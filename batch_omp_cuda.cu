/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Includes, cuda */
#include <cublas.h>
#include <cuda_runtime.h>
//#include "cublas_v2.h"

/* Number of columns & rows in dictionary */
// TODO: get as input
#define M 300  // num of Dictionary columns
#define N 50  // num of Dictionary rows
#define X 13// number of signals
/* Number of non-zero elements in signal */
int K = 4;
/* Residual error */
double epsilon = 1.0e-7;
/* Max num of iterations - assume as same as num of elements in signal */
int T = N;
/* Sign function */
double sign(double x){return (x>=0) - (x<0);}

/* Matrix indexing convention */
#define id(m, n, ld) (((n) * (ld) + (m)))

int main(int argc, char** argv)
{


	cublasStatus status;
	double *h_D, *h_X, *h_C, *c; //host memory pointers
	double *d_D = 0, *d_S = 0, *d_R = 0,*d_G = 0; //device memory pointers
	int i;

	int MX = M*X;
	int NX = M*X;
	int MN = M*N, m, n, k, q, t;
	double norm = sqrt(N), normi, normf, a, dtime,ttime = 0;

	printf("\nDictionary dimensions: N x M = %d x %d, K = %d, Number of Signals = %d", N, M, K, X);

	/* Initialize srand and clock */
	srand(time(NULL));
 	clock_t start = clock();

	/* Initialize cublas */
	status = cublasInit();
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr,"CUBLAS initialization error\n");
		return EXIT_FAILURE;
	}

	/* Initialize dictionary on host */
	h_D = (double*)malloc(MN * sizeof(h_D[0]));
	if(h_D == 0){
		fprintf(stderr, " host memory allocation error (dictionary)\n");
		return EXIT_FAILURE;
	}
	
	for(n = 0; n < N; n++){
		for(m = 0; m < M; m++){	
		a = sign(2.0*rand()/(double)RAND_MAX-1.0)/norm;
		h_D[id(m, n, M)] = a;
	 }
	}
	
	/* Create X random K-sparse signals */
	h_X = (double*)calloc(M*X, sizeof(h_X[0])); // X initialized with zeros
	
	if(h_X == 0){
		fprintf(stderr, " host memory allocation error (signal)\n");
		return EXIT_FAILURE;
	}
	for (i = 0;i < X;i++){
		for(k = 0; k < K; k++){
		a = 2.0*rand()/(double)RAND_MAX - 1.0;
		h_X[(rand()%M)+i*M] = a;}
	}

	/* Allocate solution memory on host */
	h_C = (double*)calloc(M*X, sizeof(h_C[0]));
	if(h_C == 0){
		fprintf(stderr, " host memory allocation error (solution)\n");
		return EXIT_FAILURE;
	}
	
	c = (double*)calloc(1, sizeof(c));
	if(c == 0){
		fprintf(stderr, " host memory allocation error (c)\n");
		return EXIT_FAILURE;
	}

	// start counting Host to Device MemCpy
	start = clock();

	/* Host to device data transfer: dictionary */
	status = cublasAlloc(MN, sizeof(d_D[0]),(void**)&d_D);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, " device memory allocation error (dictionary)\n");
		return EXIT_FAILURE;
	}
	
	//trasnfer the Host dictionary to Device dictionary
	status = cublasSetVector(MN, sizeof(h_D[0]),h_D, 1, d_D, 1);
	if(status != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "! device access error (write dictionary)\n");
		return EXIT_FAILURE;
	}12800
	6400
	3200
	1600
	800
	400
	200
	100
	50
	25
	13

	
	/* Host to device data transfer: signal */
	status = cublasAlloc(MX, sizeof(d_R[0]),(void**)&d_R);
	if(status != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "! device memory allocation error (signal)\n");
		return EXIT_FAILURE;
	}
	
	status = cublasSetVector(MX, sizeof(h_X[0]),h_X, 1, d_R, 1);
	if(status != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "! device access error (write signal)\n");
		return EXIT_FAILURE;
	}
	
	/*Allocate device memory for Signal Solution */
	status = cublasAlloc(NX, sizeof(d_S[0]),(void**)&d_S);
	if(status != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr, "! device memory allocation error (projected vector)\n");
		return EXIT_FAILURE;
	}
	/*Allocate Gram matrix */
		status = cublasAlloc(M*M, sizeof(d_G[0]),(void**)&d_G);
		if(status != CUBLAS_STATUS_SUCCESS){
			fprintf(stderr, "! device memory allocation error (projected vector)\n");
			return EXIT_FAILURE;
		}

	cublasDgemm('n','t', M, M,N, 1.0, d_D, M,d_D, M, 0.0, d_G, M);

	// stop counting Host to Device MemCpy
	dtime = ((double)clock()-start)/CLOCKS_PER_SEC;
	ttime = ttime + dtime;
	printf("\nTime for Host to Device MemCpy: %f(s)",dtime);

/* Encoding the signal on device*/
	for (i = 0;i<X;i++)	{
		cublasDgemv('t', M, N, 1.0, d_D, M,d_R+i*M, 1, 0.0, d_S+i*N, 1);
		status = cublasGetError();
		if(status != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "! kernel execution error (encoding)\n");
			return EXIT_FAILURE;
		}
	}




	/* Decoding the signal on device*/
	// start counting Encoding

	for (i = 0;i<X;i++)	{
		normi = cublasDnrm2 (N, d_S+i*N, 1);
		epsilon = sqrt(epsilon*normi);
		normf = normi;
		t = 0;
		while(normf > epsilon && t < T){
			//printf("\n %f",normf);
			start = clock();
			cublasDgemv('n', M, N, 1.0, d_D, M,d_S+i*N, 1, 0.0, d_R+i*M, 1);
			q = cublasIdamax (M, d_R+i*M, 1) - 1;
			cublasGetVector(1, sizeof(c),&d_R[q+i*M], 1, c, 1);
			h_C[q+i*M] = *c + h_C[q+i*M];
			dtime = ((double)clock()-start)/CLOCKS_PER_SEC;
			ttime = ttime + dtime;
			cublasDaxpy (N, -(*c), &d_D[q], M, d_S+i*N, 1);
			start = clock();
			normf = cublasDnrm2 (N, d_S+i*N, 1);
			t++;
			dtime = ((double)clock()-start)/CLOCKS_PER_SEC;
			ttime = ttime + dtime;
		}

		/*
		status = cublasGetError();
		if(status != CUBLAS_STATUS_SUCCESS){
			fprintf(stderr, "! kernel execution error (decoding)\n");
			return EXIT_FAILURE;
		*/

		a = 100.0*(normf*normf)/(normi*normi);
	//	printf("\nComputation residual error: %f",a);

		a=0; q=0; *c=0;
		epsilon=1.0e-7;
	}


	// stop counting Encoding
	//dtime = ((double)clock()-start)/CLOCKS_PER_SEC;
	//ttime = ttime + dtime;
	//printf("\nTime for Host to Device MemCpy: %f(s)",dtime);
	printf("\n Total time : %f(s) ",ttime);

	/* Check the solution */
/*
	printf("\nSolution (first column),Reference (second column):");
	getchar();  // Wait for key ...
	for(m=0; m<M; m++)
	{
		printf("\n%f\t%f\t%f\t%f", h_C[m], h_X[m],h_C[m+M],h_X[m+M]);
	}
	normi = 0; normf = 0;
	for(m=0; m<M; m++)
	{
		normi = normi + h_X[m]*h_X[m];
		normf = normf +
		(h_C[m] - h_X[m])*(h_C[m] - h_X[m]);
	}
	printf("\nSolution residual error:%f", 100.0*normf/normi);
*/
/* Memory clean up */
	free(h_D);	free(h_X);	free(h_C);
	status = cublasFree(d_D);	status = cublasFree(d_S);	status = cublasFree(d_R);
	if(status != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr,"! device memory free error\n");
		return EXIT_FAILURE;
	}
/* Shutdown */
	status = cublasShutdown();
	if(status != CUBLAS_STATUS_SUCCESS){
		fprintf(stderr,"! cublas shutdown error\n");
		return EXIT_FAILURE;
	}
	if(argc<=1 || strcmp(argv[1],"-noprompt")){
		printf("\nPress ENTER to exit...\n");
		getchar();
	}
	return EXIT_SUCCESS;
}
