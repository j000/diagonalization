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

/* global variables */
static size_t writer_count = 0; /**< counts started threads */
static pthread_t writer[4];     /**< array of threads ID */
static bool quiet = false;      /**< set to true to suppress output */
static bool symmetric = false;  /**< set to true to do calculations on symmetric matrix */

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
	if (quiet)
		return;
	print_timer(text, true);
}

void timer_no_delta(const char *text) {
	if (quiet)
		return;
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

void create_writer_thread(void *function(void *), void *args) {
	if (pthread_create(
		&writer[writer_count++],
		NULL,
		function,
		args
		) != 0) {
		fprintf(stderr, "Error creating thread. Writing in main thread.\n");
		--writer_count;
		timer("Writing");
		function(args);
	}

	timer("Starting thread");
}

/**
 * Helps with passing arguments to threads.
 */
struct thread_arguments {
	double *matrix;
	double *vector;
	size_t size;
	char *filename;
	bool binary;
};

void *write_input(void *arg_struct) {
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
			fprintf(stderr, "/!\\ INPUT MATRIX WILL NOT BE WRITTEN /!\\\n");
			return NULL;
		}
	}

	if (args->binary) {
		fwrite(&args->size, sizeof(args->size), 1, output);
		fwrite(
			args->matrix,
			sizeof(args->matrix[0]),
			args->size * args->size,
			output
		);
	} else {
		fprintf(output, "%zu\n", args->size);
		for (size_t i = 0; i < args->size; ++i) {
			for (size_t j = 0; j < args->size; ++j)
				fprintf(output, "%16.9e ", args->matrix[i * args->size + j]);
			fprintf(output, "\n");
		}
	}

	if (output != stdout)
		fclose(output);

	my_free(arg_struct);

	timer_no_delta("Writing");

	return NULL;
}

void *write_result(void *arg_struct) {
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
			fprintf(stderr, "/!\\ RESULTS WILL NOT BE WRITTEN /!\\\n");
			return NULL;
		}
	}

	if (args->binary) {
		fwrite(&args->size, sizeof(args->size), 1, output);
		fwrite(args->vector, sizeof(args->vector[0]), args->size, output);
		fwrite(
			args->matrix,
			sizeof(args->matrix[0]),
			args->size * args->size,
			output
		);
	} else {
		fprintf(output, "%zu\n", args->size);
		for (size_t i = 0; i < args->size; ++i)
			fprintf(output, "%16.9e ", args->vector[i]);
		fprintf(output, "\n");

		for (size_t i = 0; i < args->size; ++i) {
			for (size_t j = 0; j < args->size; ++j)
				fprintf(output, "%16.9e ", args->matrix[i * args->size + j]);
			fprintf(output, "\n");
		}
	}

	if (output != stdout)
		fclose(output);

	my_free(arg_struct);

	timer_no_delta("Writing");

	return NULL;
}

/* **** */

/**
 * Generate random matrix. Use Gaussian distribution with params (0, sigma).
 *
 * @param[out] matrix Pointer to generated matrix.
 * @param[in] size Matrix size.
 * @param[in] sigma Sigma in Gaussian distribution.
 * @param[in] is_symmetric If true: generated matrix will be symmetric.
 * @param[in] seed Random seed to use. Generate new seed when 0.
 */
void generate_matrix(
	double **matrix,
	size_t size,
	double sigma,
	bool is_symmetric,
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

	if (is_symmetric) {
		/* generate less random numbers
		 * we only need one triangle.
		 * also it should be faster. */
		for (size_t i = 0; i < size; i++)
			gaussian(&(*matrix)[i * (size + 1)], size - i, sigma, stream);

		/* make symmetrical */
		for (size_t i = 1; i < size; ++i)
			for (size_t j = 0; j < i; ++j)
				(*matrix)[i * size + j] = (*matrix)[j * size + i];
	} else {
		gaussian(*matrix, size * size, sigma, stream);
	}

	vslDeleteStream(&stream);
}

/* argument parsing */

const char *argp_program_bug_address =
	"https://github.com/j000/diagonalization";

/* Used by main to communicate with parse_opt. */
struct arguments {
	size_t size;
	char *input_filename;
	char *input_binary_filename;
	char *output_filename;
	char *output_binary_filename;
	bool sigma_is_inverse;
};

#define OPT_SIGMA_N 1
#define OPT_SIGMA_1_N 2
#define OPT_SYMMETRIC 3

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
	case 'b':
		arguments->input_binary_filename = arg;
		break;
	case 'o':
		arguments->output_filename = arg;
		break;
	case 'd':
		arguments->output_binary_filename = arg;
		break;
	case 's': {
			size_t tmp = 0;

			if (1 == sscanf(arg, "%zu", &tmp))
				arguments->size = tmp;
			break;
		}
	case OPT_SYMMETRIC:
		symmetric = true;
		break;
	case 'q':
		quiet = true;
		break;
	case 'i':
	case OPT_SIGMA_1_N:
		arguments->sigma_is_inverse = true;
		break;
	case OPT_SIGMA_N:
		arguments->sigma_is_inverse = false;
		break;
	default:
		return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

int main(int argc, char **argv) {
	size_t size = 10;
	double *matrix = NULL;
	double sigma = 0;
	char *input_filename = NULL;
	char *input_binary_filename = NULL;
	char *output_filename = NULL;
	char *output_binary_filename = NULL;

	assert(sizeof(MKL_INT) == sizeof(size_t));

	timer("Start");

	/* option parsing */
	{
		/* initialize defaults */
		static struct arguments arguments = {
			.size = 10,
			.input_filename = NULL,
			.input_binary_filename = NULL,
			.output_filename = NULL,
			.sigma_is_inverse = false,
		};

		/* Program documentation. */
		static char doc[] = "Matrix diagonalization." \
			"\vWhen FILE is '-', write to standard output.";

		/* The options we understand. */
		static struct argp_option options[] = {
			{ "size", 's', "N", 0, "Size of matrix to generate", 1 },
			{ "file", 'f', "FILE", 0, "Write input matrix to FILE", 2 },
			{ "binary", 'b', "FILE", 0,
				"Write input matrix in binary format to FILE",
				3 },
			{ "output", 'o', "FILE", 0,
				"Write eigenvalues and right eigenvectors to FILE", 4 },
			{ "dump-results", 'd', "FILE", 0,
				"Write eigenvalues and right eigenvectors in binary format to FILE",
				5 },
			{ "sigma-1-n", OPT_SIGMA_1_N, 0, 0, "Sigma equals 1/size", 6 },
			{ "inverse", 'i', 0, OPTION_ALIAS, 0, 0 },
			{ "sigma-n", OPT_SIGMA_N, 0, 0, "Sigma equals size (default)", 7 },
			{ "symmetric", OPT_SYMMETRIC, 0, 0, "Generate symmetric matrix",
				8 },
			{ "quiet", 'q', 0, 0, "Supress output", -1 },
			{ NULL, 'h', 0, OPTION_HIDDEN, NULL, -1 }, /* support -h */
			{ 0 }
		};

		/* Our argp parser. */
		static struct argp argp =
		{ options, parse_opt, NULL, doc, NULL, NULL, NULL };

		/* Parse our arguments; every option seen by parse_opt will
		 *      be reflected in arguments. */
		argp_parse(&argp, argc, argv, 0, 0, &arguments);

		input_filename = arguments.input_filename;
		input_binary_filename = arguments.input_binary_filename;
		output_filename = arguments.output_filename;
		output_binary_filename = arguments.output_binary_filename;
		if (arguments.size > 0)
			size = arguments.size;
		if (arguments.sigma_is_inverse == true)
			sigma = 1. / size;
		else
			sigma = size;

		if (!quiet) {
			printf("Generating ");
			if (symmetric)
				printf("symmetric ");
			printf("matrix %zux%zu.\n", size, size);
			if (arguments.sigma_is_inverse == true)
				printf("Sigma: %.6f\n", sigma);
			if (input_filename != NULL) {
				printf("Writing input matrix into ");
				if (strcmp(input_filename, "-") != 0)
					printf("file %s.\n", input_filename);
				else
					printf("stdout.\n");
			}
			if (input_binary_filename != NULL) {
				printf("Writing input matrix in binary format into ");
				if (strcmp(input_binary_filename, "-") != 0)
					printf("file %s.\n", input_binary_filename);
				else
					printf("stdout.\n");
			}
			if (output_filename != NULL) {
				printf("Writing resutls into ");
				if (strcmp(output_filename, "-") != 0)
					printf("file %s.\n", output_filename);
				else
					printf("stdout.\n");
			}
			if (output_binary_filename != NULL) {
				printf("Writing resutls in binary format into ");
				if (strcmp(output_binary_filename, "-") != 0)
					printf("file %s.\n", output_binary_filename);
				else
					printf("stdout.\n");
			}
			printf("\n");
		}
	}

	assert(size * size * sizeof(*matrix) <= SIZE_MAX);

	/* mkl_disable_fast_mm(); */
	mkl_peak_mem_usage(MKL_PEAK_MEM_ENABLE);

	timer("Parsing options");

	generate_matrix(&matrix, size, size, symmetric, 0);

	timer("Generating");

	if (input_filename != NULL) {
		struct thread_arguments *args = my_calloc(1, sizeof(*args));

		args->matrix = matrix;
		args->size = size;
		args->filename = input_filename;

		/* create writer thread */
		create_writer_thread(write_input, args);
	}

	if (input_binary_filename != NULL) {
		struct thread_arguments *args = my_calloc(1, sizeof(*args));

		args->matrix = matrix;
		args->size = size;
		args->filename = input_binary_filename;
		args->binary = true;

		/* create writer thread */
		create_writer_thread(write_input, args);
	}

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
		fprintf(stderr, "dfeast_syev error: %" MKL_FORMAT ".\n", info);
		fprintf(
			stderr,
			"https://software.intel.com/en-us/mkl-developer-reference-c-extended-eigensolver-output-details\n"
		);
		return 1;
	}

	if (M < (MKL_INT)size)
		fprintf(stderr, "/!\\ NOT ENOUGH EIGENVALUES FOUND /!\\\n");

	if (!quiet)
		printf("%" MKL_FORMAT " eigenvalues found.\n", M);

	timer("Computing");

	if (output_filename != NULL) {
		struct thread_arguments *args = my_calloc(1, sizeof(*args));

		args->size = size;
		args->matrix = X;
		args->vector = E;
		args->filename = output_filename;

		/* create writer thread */
		create_writer_thread(write_result, args);
	}

	if (output_binary_filename != NULL) {
		struct thread_arguments *args = my_calloc(1, sizeof(*args));

		args->size = size;
		args->matrix = X;
		args->vector = E;
		args->filename = output_binary_filename;
		args->binary = true;

		/* create writer thread */
		create_writer_thread(write_result, args);
	}

	/* wait for threads */
	for (size_t i = 0; i < writer_count; ++i)
		pthread_join(writer[i], NULL);

	timer("Waiting for threads");

	/* **** */

	{
		char normal = 'N';
		double one = 1.0;
		double zero = 0.0;
		double *tmp = my_calloc(size * size, sizeof(*tmp));
		/* create matrix with eigenvalues on diagonal */
		/* ident = I*E */
		double *ident = my_calloc(size * size, sizeof(*ident));

		for (size_t i = 0; i < size; ++i)
			ident[i * size + i] = E[i];

		/* tmp = P*D */
		dgemm(
			&normal,          /* 'N', non-transposed */
			&normal,          /* 'N', non-transposed */
			(MKL_INT *)&size, /* Number of rows in destination */
			(MKL_INT *)&size, /* Number of columns in matrix #1 */
			(MKL_INT *)&size, /* Number of columns in destination */
			&one,             /* alpha = 1.0 */
			X,                /* Source #1 for GEMM */
			(MKL_INT *)&size, /* Leading dimension of Source 1 */
			ident,            /* Source #2 for GEMM */
			(MKL_INT *)&size, /* Leading dimension of Source 2 */
			&zero,            /* beta = 0.0 */
			tmp,              /* out: Destination */
			(MKL_INT *)&size  /* Leading dimension of Destination */
		);

		/* ident = A*P */
		dgemm(
			&normal,          /* 'N', non-transposed */
			&normal,          /* 'N', non-transposed */
			(MKL_INT *)&size, /* Number of rows in destination */
			(MKL_INT *)&size, /* Number of columns in matrix #1 */
			(MKL_INT *)&size, /* Number of columns in destination */
			&one,             /* alpha = 1.0 */
			matrix,           /* Source #1 for GEMM */
			(MKL_INT *)&size, /* Leading dimension of Source 1 */
			X,                /* Source #2 for GEMM */
			(MKL_INT *)&size, /* Leading dimension of Source 2 */
			&zero,            /* beta = 0.0 */
			ident,            /* out: Destination */
			(MKL_INT *)&size  /* Leading dimension of Destination */
		);

		double max = 0.;
		double error = 0.;

		for (size_t i = 0; i < size; ++i)
			for (size_t j = 0; j < size; ++j) {
				double t = abs(tmp[i * size + j] - ident[i * size + j]);
				if (t > max)
					max = t;
				error += t;
			}

		my_free(ident);
		my_free(tmp);

		printf("Max error: %16.9E\n", max);
		printf("Sum of errors: %16.9E\n", error);
		timer("Verifying");
	}

	/* **** */

	mkl_free_buffers();
	my_free(res);
	my_free(E);
	my_free(X);
	my_free(matrix);

#ifdef _DEBUG
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
#endif /* ifdef _DEBUG */
}
