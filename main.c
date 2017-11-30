#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#include <math.h>

#include <sys/random.h>

#define MKL_ILP64      /* in order to use size_t instead of MKL_INT */
#define MKL_INT size_t /* this tells MKL about user's MKL_INT type */

#include <mkl_solvers_ee.h>
#include <mkl_vsl.h>
#include <mkl.h>

/* **** */

/**
 * Calloc wrapper: shows message and exits on error.
 * @param n number of elements to allocate
 * @param size element size in bytes
 */
void *my_calloc(size_t n, size_t size) {
	void *tmp = calloc(n, size);

	if (tmp != NULL)
		return tmp;
	fprintf(
		stderr,
		"Calloc failed (%ld x %ld bytes): %s. Aborting.\n",
		n,
		size,
		strerror(errno)
	);
	exit(EXIT_FAILURE);
}

void gaussian(double *buf, const size_t size, const double sigma) {
	static VSLStreamStatePtr stream = NULL;
	static int_fast64_t seed = 0;
	MKL_INT status;

	if (seed == 0)      /* get random seed */
		if (getrandom(&seed, sizeof(seed), 0) == -1) {
			fprintf(stderr, "getrandom() failed.\n");
			exit(EXIT_FAILURE);
		}

	if (stream == NULL) /* initialize stream */
		vslNewStream(&stream, VSL_BRNG_MT19937, seed);

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
			"Random number generator failed with error %ld\n",
			status
		);
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char **argv) {
	size_t size = 11;
	double *matrix;

	matrix = my_calloc(size * size, sizeof(*matrix));

	{
		double sigma = sqrt(size);

		/* gaussian(matrix, size*size, sqrt(size)); */
		for (size_t i = 0; i < size; i++)
			gaussian(&matrix[i * (size + 1)], size - i, sigma);
	}

	/* make symmetrical */
	for (size_t i = 1; i < size; ++i)
		for (size_t j = 0; j < i; ++j)
			matrix[i * size + j] = matrix[j * size + i];

	/* print */
	for (size_t i = 0; i < size; ++i) {
		for (size_t j = 0; j < size; ++j)
			printf("%7.3f ", matrix[i * size + j]);
		printf("\n");
	}

	free(matrix);
}
