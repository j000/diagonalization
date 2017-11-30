#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <assert.h>

#include <math.h>

#include <sys/random.h>

#define MKL_ILP64      /* in order to use size_t instead of MKL_INT */
#define MKL_INT int_fast64_t
#define MKL_UINT uint_fast64_t
#define MKL_FORMAT "ld"

#include <mkl_solvers_ee.h>
#include <mkl_vsl.h>
#include <mkl.h>

/* **** */

/**
 * Calloc wrapper: shows message and exits on error.
 *
 * @param n number of elements to allocate
 * @param size element size in bytes
 */
void *my_calloc(size_t n, size_t size) {
	void *tmp = calloc(n, size);
	/* void *tmp = mkl_calloc(n, size, 64); */

	if (tmp != NULL)
		return tmp;
	fprintf(
		stderr,
		"Calloc failed (%ld x %ld = %ldkB): %s. Aborting.\n",
		n,
		size,
		n * size / 1024,
		strerror(errno)
	);
	exit(EXIT_FAILURE);
}

void timer(const char* text) {
	static struct timespec start = { 0 };
	static struct timespec last = { 0 };
	static struct timespec tmp;

	if (start.tv_sec == 0) {
		if (clock_gettime(CLOCK_MONOTONIC_RAW, &start) == -1)
			return;
		last = start;
	}

	clock_gettime(CLOCK_MONOTONIC_RAW, &tmp);
	double difference = (tmp.tv_sec - start.tv_sec) + (tmp.tv_nsec - start.tv_nsec) * 1e-9;
	printf("TIMER: %7.2lfs ", difference);
	difference = (tmp.tv_sec - last.tv_sec) + (tmp.tv_nsec - last.tv_nsec) * 1e-9;
	printf("(%+7.2lfs) - %s\n", difference, text);
	last = tmp;
}

/**
 * Random number generator of Gaussian (normal) distribution.
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
			"Random number generator failed with error %" MKL_FORMAT "\n",
			status
		);
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char **argv) {
	size_t size = 2000;
	double *matrix;
	static VSLStreamStatePtr stream = NULL;
	static int_fast32_t seed = 0;

	assert(size * size * sizeof(*matrix) <= SIZE_MAX);
	assert(sizeof(MKL_INT) == sizeof(size_t));

	timer("Start");

	/* mkl_disable_fast_mm(); */
	mkl_peak_mem_usage(MKL_PEAK_MEM_ENABLE);

	if (seed == 0) /* get random seed */
		if (getrandom(&seed, sizeof(seed), 0) == -1) {
			fprintf(stderr, "getrandom() failed.\n");
			/* manually read /dev/urandom */
			FILE *urandom = fopen("/dev/urandom", "r");
			if (urandom == NULL	|| fread(&seed, sizeof(seed), 1, urandom) < 1) {
				fprintf(stderr, "Something went wrong while opening /dev/urandom\n");
				exit(EXIT_FAILURE);
			}
		}

	/* initialize random stream */
	{
		MKL_INT status = vslNewStream(&stream, VSL_BRNG_MT19937, seed);

		if (status != VSL_STATUS_OK) {
			fprintf(
				stderr,
				"Cannot create new random stream. Error %" MKL_FORMAT "\n",
				status
			);
			exit(EXIT_FAILURE);
		}
	}

	timer("Generating");

	matrix = my_calloc(size * size, sizeof(*matrix));

	/* generate less random numbers
	 * we only need one triangle.
	 * Also it should be faster than: */
	/* gaussian(matrix, size*size, sqrt(size)); */
	for (size_t i = 0; i < size; i++)
		gaussian(&matrix[i * (size + 1)], size - i, size, stream);

	/* make symmetrical */
	for (size_t i = 1; i < size; ++i)
		for (size_t j = 0; j < i; ++j)
			matrix[i * size + j] = matrix[j * size + i];

	/* print */
	/* for (size_t i = 0; i < size; ++i) { */
	/*  for (size_t j = 0; j < size; ++j) */
	/*      printf("%7.3f ", matrix[i * size + j]); */
	/*  printf("\n"); */
	/* } */

	timer("Computing");

	/* **** */

	/* FEAST variables */
	MKL_INT fpm[128] = { 0 };                       /* parameters */
	const char type = 'F';                          /* full matrix */
	double Emax = size * size;                         /* search inteval [ Emin ; Emax ] */
	double Emin = -Emax;
	double epsilon = 0;
	MKL_INT loops = 0;
	MKL_INT M0 = size;                               /* guessed subspace dimension */
	MKL_INT M = 0;                                   /* number of eigenvalues found */
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
		(MKL_INT*)&size,                                      /* size of the problem */
		matrix,                                     /* matrix */
		(MKL_INT*)&size,                                      /* size of matrix */
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
		printf("dfeast_syev error: %" MKL_FORMAT "\n", info);
		printf(
			"https://software.intel.com/en-us/mkl-developer-reference-c-extended-eigensolver-output-details\n"
		);
		return 1;
	}
	printf("%" MKL_FORMAT "\n", M);

	timer("Done");
	/* **** */

	vslDeleteStream(&stream);
	mkl_free_buffers();
	mkl_finalize();
	free(matrix);
	/* mkl_free(matrix); */

	{
		int N_AllocatedBuffers;
		size_t AllocatedBytes = mkl_mem_stat(&N_AllocatedBuffers);

		if (AllocatedBytes > 0) {
			printf("MKL memory leak!\n");
			printf(
				"After mkl_free_buffers there are %ld bytes in %d buffers\n",
				AllocatedBytes,
				N_AllocatedBuffers
			);
		}
	}
}
