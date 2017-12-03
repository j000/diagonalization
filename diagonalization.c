#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <pthread.h>
#include <argp.h>
#include <time.h>
#include <errno.h>
#include <assert.h>
#include <stdbool.h>

#include <math.h>

#include <sys/random.h>

#define MKL_ILP64 /* in order to use size_t instead of MKL_INT */
#define MKL_INT int_fast64_t
#define MKL_UINT uint_fast64_t
#define MKL_FORMAT PRIdFAST64

#include <mkl_solvers_ee.h>
#include <mkl_vsl.h>
#include <mkl.h>

/* **** */

/**
 * Calloc wrapper: shows message and exits on error.
 *
 * @param n Number of elements to allocate.
 * @param size Element size in bytes.
 * @return Pointer to allocated memory.
 */
void *my_calloc(size_t n, size_t size) {
#ifndef USE_MKL_ALLOC
	void *tmp = calloc(n, size);
#else /* ifndef USE_MKL_ALLOC */
	void *tmp = mkl_calloc(n, size, 64);
#endif /* ifndef USE_MKL_ALLOC */

	if (tmp != NULL)
		return tmp;
	fprintf(
		stderr,
		"Calloc failed (%zu x %zu = %zukB): %s. Aborting.\n",
		n,
		size,
		n * size / 1024,
		strerror(errno)
	);
	exit(EXIT_FAILURE);
}

/**
 * Free wrapper. Because we have different allocs, it's easier to have a wrapper.
 *
 * @param[in] mem Pointer to be freed.
 */
void my_free(void *mem) {
#ifndef USE_MKL_ALLOC
	free(mem);
#else /* ifndef USE_MKL_ALLOC */
	mkl_free(mem);
#endif /* ifndef USE_MKL_ALLOC */
}

void print_timer(const char *text, bool reset_delta) {
	static struct timespec start = { 0 };
	static struct timespec last = { 0 };
	static struct timespec tmp;

	if (start.tv_sec == 0) {
		if (clock_gettime(CLOCK_MONOTONIC_RAW, &start) == -1)
			return;
		last = start;
		return;
	}

	clock_gettime(CLOCK_MONOTONIC_RAW, &tmp);

	double total = (tmp.tv_sec - start.tv_sec) + (tmp.tv_nsec - start.tv_nsec) *
		1e-9;
	double delta = (tmp.tv_sec - last.tv_sec) + (tmp.tv_nsec - last.tv_nsec) *
		1e-9;

	printf("TIMER: %6.2lfs %25s:%6.2lfs\n", total, text, delta);

	if (reset_delta)
		last = tmp;
}

void timer(const char *text) {
	print_timer(text, true);
}

void timer_no_delta(const char *text) {
	print_timer(text, false);
}

/**
 * Random number generator of Gaussian (normal) distribution.
 *
 * @param[out] buf Pointer to where to store random numbers.
 * @param[in] size How many numbers to generate.
 * @param[in] sigma Sigma in Gaussian distribution.
 * @param stream Random numbers stream to use.
 */
void gaussian(
	double *buf,
	const size_t size,
	const double sigma,
	VSLStreamStatePtr stream
) {
	MKL_INT status;

	/* VSL_RNG_METHOD_GAUSSIAN_ICDF: */
	/* Random number generator of normal (Gaussian) distribution with parameters a and s */
	/* https://software.intel.com/en-us/node/590423 */
	status = vdRngGaussian(
		VSL_RNG_METHOD_GAUSSIAN_ICDF,
		stream,
		size,
		buf,
		0.0,
		sigma
		);
	if (status != VSL_STATUS_OK) {
		fprintf(
			stderr,
			"Random number generator failed with error %" MKL_FORMAT ".\n",
			status
		);
		exit(EXIT_FAILURE);
	}
}

/* thread stuff */

/**
 * Helps with passing arguments to threads.
 */
struct thread_arguments {
	double *matrix;
	size_t size;
	char *filename;
};

void *write_matrix(void *arg_struct) {
	struct thread_arguments *args = (struct thread_arguments *)arg_struct;
	FILE *output;

	if (args->filename == NULL)
		return NULL;

	if (strcmp(args->filename, "-") == 0) {
		output = stdout;
	} else {
		output = fopen(args->filename, "w");
		if (output == NULL) {
			fprintf(stderr, "Uname to open %s for writing.\n", args->filename);
			fprintf(stderr, "INPUT MATRIX WILL NOT BE WRITTEN!\n");
			return NULL;
		}
	}

	fprintf(output, "%zu\n", args->size);
	for (size_t i = 0; i < args->size; ++i) {
		for (size_t j = 0; j < args->size; ++j)
			fprintf(output, "%16.9e ", args->matrix[i * args->size + j]);
		fprintf(output, "\n");
	}

	if (output != stdout)
		fclose(output);
	my_free(arg_struct);

	timer_no_delta("Writing");

	return NULL;
}

/* **** */

void generate_symmetric_matrix(
	double **matrix,
	size_t size,
	double sigma,
	int_fast32_t seed
) {
	VSLStreamStatePtr stream = NULL;

	if (seed == 0) /* get random seed */
		if (getrandom(&seed, sizeof(seed), 0) == -1) {
			fprintf(stderr, "getrandom() failed.\n");
			/* manually read /dev/urandom */
			FILE *urandom = fopen("/dev/urandom", "r");
			if (urandom == NULL	|| fread(&seed, sizeof(seed), 1, urandom) < 1) {
				fprintf(
					stderr,
					"Something went wrong while opening /dev/urandom.\n"
				);
				exit(EXIT_FAILURE);
			}
			fclose(urandom);
		}

	/* initialize random stream */
	{
		MKL_INT status = vslNewStream(&stream, VSL_BRNG_MT19937, seed);

		if (status != VSL_STATUS_OK) {
			fprintf(
				stderr,
				"Cannot create new random stream. Error %" MKL_FORMAT ".\n",
				status
			);
			exit(EXIT_FAILURE);
		}
	}

	*matrix = my_calloc(size * size, sizeof(**matrix));

	/* generate less random numbers
	 * we only need one triangle.
	 * Also it should be faster than: */
	/* gaussian(matrix, size*size, sigma); */
	for (size_t i = 0; i < size; i++)
		gaussian(&(*matrix)[i * (size + 1)], size - i, sigma, stream);

	/* make symmetrical */
	for (size_t i = 1; i < size; ++i)
		for (size_t j = 0; j < i; ++j)
			(*matrix)[i * size + j] = (*matrix)[j * size + i];

	vslDeleteStream(&stream);
}

/* argument parsing */

const char *argp_program_bug_address =
	"https://github.com/j000/diagonalization";

/* Used by main to communicate with parse_opt. */
struct arguments {
	size_t size;
	char *input_filename;
};

/* Parse a single option. */
static error_t parse_opt(int key, char *arg, struct argp_state *state) {
	/* Get the input argument from argp_parse, which we
	 * know is a pointer to our arguments structure. */
	struct arguments *arguments = state->input;

	switch (key) {
	case 'h':
		argp_state_help(state, state->out_stream, ARGP_HELP_STD_HELP);
		break;
	case 'f':
		arguments->input_filename = arg;
		break;
	case 's': {
			size_t tmp = 0;

			if (1 == sscanf(arg, "%zu", &tmp))
				arguments->size = tmp;
			break;
		}
	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

int main(int argc, char **argv) {
	size_t size = 10;
	double *matrix = NULL;
	pthread_t writer;
	char *filename = NULL;

	assert(sizeof(MKL_INT) == sizeof(size_t));

	timer("Start");

	/* option parsing */
	{
		/* initialize defaults */
		static struct arguments arguments = {
			.size = 10,
			.input_filename = NULL,
		};

		/* Program documentation. */
		static char doc[] = "Diagonalizacja macierzy" \
			"\vWhen FILE is '-', write to standard output.";

		/* The options we understand. */
		static struct argp_option options[] = {
			{ "size", 's', "N", 0, "Size of matrix to generate", 0 },
			{ "file", 'f', "FILE", 0, "Write input matrix to FILE", 1 },
			{ NULL, 'h', 0, OPTION_HIDDEN, NULL, -1 }, /* support -h */
			{ 0 }
		};

		/* Our argp parser. */
		static struct argp argp =
		{ options, parse_opt, NULL, doc, NULL, NULL, NULL };

		/* Parse our arguments; every option seen by parse_opt will
		 *      be reflected in arguments. */
		argp_parse(&argp, argc, argv, 0, 0, &arguments);

		filename = arguments.input_filename;
		if (arguments.size > 0)
			size = arguments.size;

		printf("\n");
		printf("Generating matrix %zux%zu.\n", size, size);
		if (filename != NULL) {
			printf("Writing input matrix into ");
			if (strcmp(filename, "-") != 0)
				printf("file %s.\n", filename);
			else
				printf("stdout.\n");
		}
		printf("\n");
	}

	assert(size * size * sizeof(*matrix) <= SIZE_MAX);

	/* mkl_disable_fast_mm(); */
	mkl_peak_mem_usage(MKL_PEAK_MEM_ENABLE);

	timer("Parsing options");

	generate_symmetric_matrix(&matrix, size, size, 0);

	timer("Generating");

	if (filename != NULL) {
		struct thread_arguments *args = my_calloc(1, sizeof(*args));

		args->matrix = matrix;
		args->size = size;
		args->filename = filename;

		/* create writer thread */
		if (pthread_create(&writer, NULL, write_matrix, args) != 0) {
			fprintf(stderr, "Error creating thread. Writing in main thread.\n");
			timer("Writing");
			write_matrix(args);
			/* exit(EXIT_FAILURE); */
		}
	}

	timer("Starting thread");

	/* **** */

	/* FEAST variables */
	MKL_INT fpm[128] = { 0 };                       /* parameters */
	const char type = 'F';                          /* full matrix */
	double Emax = size * size;                      /* search inteval [ Emin ; Emax ] */
	double Emin = -Emax;
	double epsilon = 0;
	MKL_INT loops = 0;
	MKL_INT M0 = size;                              /* guessed subspace dimension */
	MKL_INT M = 0;                                  /* number of eigenvalues found */
	double *E = my_calloc(size, sizeof(*E));        /* eigenvalues */
	double *X = my_calloc(size * size, sizeof(*X)); /* eigenvectors */
	double *res = my_calloc(size, sizeof(*res));    /* residual */
	MKL_INT info = 0;                               /* error */

	feastinit(fpm);                                 /* get default parameters */
	fpm[2] = 10;                                    /* stopping criteria: 10^(-fpm[2]) */
	fpm[26] = 1;                                    /* check input matrices */

	/* call solver */
	dfeast_syev(
		&type,                                      /* matrix type */
		(MKL_INT *)&size,                           /* size of the problem */
		matrix,                                     /* matrix */
		(MKL_INT *)&size,                           /* size of matrix */
		fpm,                                        /* parameters */
		&epsilon,                                   /* out: relative error */
		&loops,                                     /* out: number of loops executed */
		&Emin,                                      /* search interval */
		&Emax,
		&M0,                                        /* guessed dimensions */
		E,                                          /* out: M eigenvalues */
		X,                                          /* out: M eigenvectors */
		&M,                                         /* out: eigenvalues found */
		res,                                        /* out: M relative resiual vector */
		&info                                       /* out: error code */
	);

	if (info != 0) {
		printf("dfeast_syev error: %" MKL_FORMAT ".\n", info);
		printf(
			"https://software.intel.com/en-us/mkl-developer-reference-c-extended-eigensolver-output-details\n"
		);
		return 1;
	}
	printf("%" MKL_FORMAT "\n", M);

	timer("Computing");

	/* **** */

	mkl_free_buffers();
	mkl_finalize();
	my_free(matrix);

	{
		int N_AllocatedBuffers;
		size_t AllocatedBytes = mkl_mem_stat(&N_AllocatedBuffers);

		if (AllocatedBytes > 0) {
			printf("MKL memory leak!\n");
			printf(
				"After mkl_free_buffers there are %ld bytes in %d buffers.\n",
				AllocatedBytes,
				N_AllocatedBuffers
			);
		}
	}
	pthread_join(writer, NULL);
}
