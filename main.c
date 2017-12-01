#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <pthread.h>
#include <getopt.h>
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

struct arguments {
	double *matrix;
	size_t size;
	char *filename;
};

void *write_matrix(void *arg_struct) {
	struct arguments *args = (struct arguments *)arg_struct;
	FILE *output;

	if (args->filename == NULL) {
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

	if (args->filename != NULL)
		fclose(output);
	my_free(arg_struct);

	timer_no_delta("Writing");

	return NULL;
}

int main(int argc, char **argv) {
	size_t size = 10;
	double *matrix;
	VSLStreamStatePtr stream = NULL;
	int_fast32_t seed = 0;
	pthread_t writer;
	char *filename = NULL;

	timer("Start");

	assert(size * size * sizeof(*matrix) <= SIZE_MAX);
	assert(sizeof(MKL_INT) == sizeof(size_t));

	/* option parsing */
	{
		extern int opterr;

		opterr = 0;
		while (1) {
			/* getopt_long stores the option index here. */
			int option_index = 0;
			char c;
			static struct option long_options[] = {
				{ "file", required_argument, 0, 'f' },
				{ "size", required_argument, 0, 's' },
				{ 0, 0, 0, 0 }
			};

			c = getopt_long(
				argc,
				argv,
				"f:s:",
				long_options,
				&option_index
				);

			/* Detect the end of the options. */
			if (c == -1)
				break;

			switch (c) {
			case 'f': {
					size_t length = strlen(optarg);

					if (filename != NULL)
						my_free(filename);
					filename = my_calloc(length + 1, sizeof(*filename));
					strncpy(filename, optarg, length);
					break;
				}

			case 's': {
					size_t tmp = 0;

					if (1 == sscanf(optarg, "%zu", &tmp))
						size = tmp;
					break;
				}

			case '?':
				fprintf(stderr, "Unknown option %d\n", optopt);
				exit(EXIT_FAILURE);
				break;

			default:
				fprintf(stderr, "Unexpected result from getopt_long\n");
				exit(EXIT_FAILURE);
			}
		}

		/* Print any remaining command line arguments (not options). */
		if (optind < argc) {
			fprintf(stderr, "Unknown arguments: ");
			while (optind < argc)
				printf("%s\n", argv[optind++]);
			exit(EXIT_FAILURE);
		}

		printf("\n");
		printf("Generating matrix %zux%zu.\n", size, size);
		if (filename != NULL)
			printf("Writing input matrix into file %s.\n", filename);
		printf("\n");
	}

	/* mkl_disable_fast_mm(); */
	mkl_peak_mem_usage(MKL_PEAK_MEM_ENABLE);

	timer("Parsing options");

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

	matrix = my_calloc(size * size, sizeof(*matrix));

	/* generate less random numbers
	 * we only need one triangle.
	 * Also it should be faster than: */
	/* gaussian(matrix, size*size, sqrt(size)); */
	for (size_t i = 0; i < size; i++)
		gaussian(&matrix[i * (size + 1)], size - i, size, stream);

	timer("Generating");

	/* make symmetrical */
	for (size_t i = 1; i < size; ++i)
		for (size_t j = 0; j < i; ++j)
			matrix[i * size + j] = matrix[j * size + i];

	{
		struct arguments *args = my_calloc(1, sizeof(*args));

		args->matrix = matrix;
		args->size = size;
		args->filename = filename;

		/* create a second thread */
		if (pthread_create(&writer, NULL, write_matrix, args) != 0) {
			fprintf(stderr, "Error creating thread. Writing locally.\n");
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

	vslDeleteStream(&stream);
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
