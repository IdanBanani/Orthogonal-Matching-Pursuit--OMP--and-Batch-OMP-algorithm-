/*Naive CPU Implementation */

/* Includes, system */
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

#define M 300 // Number of columns in dictionary*/
#define N 50 // Number of rows in dictionary */
int K = 4; //Number of non-zero elements in signal - the sparsity 
int P = 25; //Number of signals
double epsilon = 1.0e-7; // Residual error 
int numOfIterations = N; /* Max num of iterations - assume as same as num of elements in signal */

double sign(double X){return (X>=0) - (X<0);} // Sign function 

int main(int argc, char** argv)
{
int n, m, k, iter, q, i;
double normi, normf;  // residual l2 norm
double tmp , norm = sqrt(N), decodeTime;
gsl_matrix *D;  // A random dictionary used for encoding the sparse signal  NxM
gsl_matrix *X;  // Sparse info signal (encoder input) MxP
gsl_matrix *Z;	// Evaluated Sparse info signal (decoder output)  MxP
gsl_matrix *R;  // Residual error matrix MxP
gsl_matrix *Y;	// Sparse representation of signal (encoder output) NxP
gsl_vector_view v;  //A vector view is a temporary object, stored on the stack, 
					//which can be used to operate on a subset of vector elements.
gsl_vector_view v2;

clock_t start; //for measuring performance

printf("\nNumOfSignals=%d, Dictionary is:NxM=%dx%d, Signal sparsity K=%d",P, N, M, K);

/* Initiallize D as a Bernoulli random dictionary */
D = gsl_matrix_alloc (N, M);
for(m=0; m<M; m++)
	{
	for(n=0; n<N; n++)
	 {
		tmp=sign(2.0*rand()/(double)RAND_MAX-1.0)/norm;
		gsl_matrix_set (D, n, m, tmp);    //D[n,m]=tmp
	 }
   }

/* Create P random K-sparse info signals matrix */
X = gsl_matrix_calloc (M,P);
for (i = 0;i < P;i++){
	for(k=0; k<K; k++)
		{
		//put random values at k random positions in each signal
		gsl_matrix_set(X, rand()%M, i, 2.0*rand()/(float)RAND_MAX - 1.0); 
		}
	}
 
/* Allocate memory for solution (evaluated signal) */
Z = gsl_matrix_calloc (M,P); 

/* Allocate memory for residual matrix */
R = gsl_matrix_calloc (M,P);

/* Allocate memory for the encoded signal matrix (its representation) */
Y = gsl_matrix_calloc (N,P);


/* Encoding the signal (X to Y) */
gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1, D, X, 0, Y); // Y = D*X
 
/* Decoding the signal */
srand( (unsigned int)time(NULL)); //Initialize srand
start = clock();  //Initialize  clock


for (i = 0;i < P;i++){ //repeat for each signal
	v = gsl_matrix_column(Y, i);
	normi = gsl_blas_dnrm2(&v.vector); // ||y|| (L2 norm)
	epsilon = sqrt(epsilon * normi); //update epsilon with projection value
	normf = normi;					// Init conditional l2 norm
	iter= 0;						//iteration number

	/*iterate till the computational error is small enough*/
	while(normf > epsilon && iter < numOfIterations)
	{
		gsl_blas_dgemm (CblasTrans, CblasNoTrans, 1, D, Y, 0, R);// R=D'*Y
		v = gsl_matrix_column(R, i);
		q = gsl_blas_idamax(&v.vector); //index of max element in R i'th signal
		tmp = gsl_vector_get(&v.vector, q); //the  max element in R i'th signal
		v = gsl_matrix_column(Z, i);        //get the i'th signal of Z
		gsl_vector_set(&v.vector, q, gsl_vector_get(&v.vector, q)+tmp); // Z[q,i]=Z[q,i]+ max element in R i'th signal
		v = gsl_matrix_column(D, q); // choose the dictrionary's atom (coloum) with 
									//the index of largest element in R i'th signal
		v2=	gsl_matrix_column(Y, i); //get the i'th signal of Z						
        gsl_blas_daxpy(-tmp,&v.vector,&v2.vector); // y = y-tmp*v
		normf = gsl_blas_dnrm2(&v2.vector); // update||y|| (L2 norm)
		iter++;
	}
	
	tmp = 100.0*(normf*normf)/(normi*normi); // residual error
	//printf("\nComputation residual error: %f",tmp);
	tmp=0;
	epsilon=1.0e-7;
	
}
	
decodeTime = ((double)clock()-start)/CLOCKS_PER_SEC;
printf("\nTotal Time: %f (s)", decodeTime);

/* Check the solution (evaluated signal) against the original signal */
//repeat for each signal
for (i = 0;i < P;i++){
	v = gsl_matrix_column(X, i);
	v2 = gsl_matrix_column(Z, i);
/*
	printf("\nSolution (first column),Reference (second column):");
	getchar(); // wait for pressing a key
	for(m=0; m<M; m++)
	{	
		printf("\n%.6f\t%.6f", gsl_vector_get(&v.vector, m),gsl_vector_get(&v2.vector, m));
	}


	normi = gsl_blas_dnrm2(&v.vector); // ||x||
	gsl_blas_daxpy(-1.0, &v.vector, &v2.vector); // z = z-x
	normf = gsl_blas_dnrm2(&v2.vector); // ||z|| (L2 norm)
	tmp = 100.0*(normf*normf)/(normi*normi); //final error
	printf("\nSolution residual error: %f\n",tmp);
	*/

}

/* Memory clean up and shutdown*/
gsl_matrix_free(Y); gsl_matrix_free(R);
gsl_matrix_free(Z); gsl_matrix_free(X);
gsl_matrix_free(D);	getchar();
return EXIT_SUCCESS;
}
