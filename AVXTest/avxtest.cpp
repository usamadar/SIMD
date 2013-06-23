#include <iostream>
#include <xmmintrin.h>
#include <conio.h>
#include <Windows.h>
#include <immintrin.h>

#define ARRAY_SIZE 160000

double PCFreq = 0.0;
__int64 CounterStart = 0;

void StartCounter();
double GetCounter();

void ComputeArrayCPlusPlus(float* pArray1, float* pArray2, float* pResult, int nSize);
void ComputeArrayCPlusPlusSSE(float* pArray1, float* pArray2, float* pResult, int nSize) ;
void ComputeArrayCPlusPlusAVX(float* pArray1, float* pArray2, float* pResult, int nSize); 

void print_float(float a[]);
float RandomFloat(float a, float b);	


void main(void) {
		// declare 16 byte aligned arrays for SSE
	__declspec(align(16)) float src_sse[ARRAY_SIZE];
	__declspec(align(16)) float src2_sse[ARRAY_SIZE];
	__declspec(align(16)) float result_sse[ARRAY_SIZE];
		// declare 32 byte aligned arrays for AVX
	__declspec(align(32)) float src_avx[ARRAY_SIZE];
	__declspec(align(32)) float src2_avx[ARRAY_SIZE];
	__declspec(align(32)) float result_avx[ARRAY_SIZE];
	int i;

	for (i=0; i< ARRAY_SIZE; i++) {
		src_sse[i] = src_avx[i] = ((float) rand()) / (float) RAND_MAX;
		src2_sse[i] = src2_avx[i] = ((float) rand()) / (float) RAND_MAX;
	}

	StartCounter();
	ComputeArrayCPlusPlus(src_sse, src2_sse, result_sse, ARRAY_SIZE);
	std::cout << "It took " << GetCounter() << "ms to compute without SIMD \n";
	

	std::cout << "Press Any Key\n";

	_getch();

	StartCounter();
	ComputeArrayCPlusPlusSSE(src_sse, src2_sse, result_sse, ARRAY_SIZE);
	std::cout << "It took " << GetCounter() << "ms to compute with SSE \n";
	
	//print_float(result);
	std::cout << "Press Any Key\n";
	_getch();

	StartCounter();
	ComputeArrayCPlusPlusAVX(src_avx, src2_avx, result_avx, ARRAY_SIZE);
	std::cout << "It took " << GetCounter() << "ms to compute with AVX \n";
	
	//print_float(result);
	std::cout << "Press Any Key";
	_getch();

}
void print_float(float a[]) {
	int i;
	for (i=0;i < ARRAY_SIZE; i++)
	{
		printf("%f\t", a[i]);
	}
}
void ComputeArrayCPlusPlusSSE(
	float* pArray1,                   // [in] first source array
	float* pArray2,                   // [in] second source array
	float* pResult,                   // [out] result array
	int nSize)                        // [in] size of all arrays
{
	int nLoop = nSize/ 4;

	__m128 m1, m2, m3, m4;

	__m128* pSrc1 = (__m128*) pArray1;
	__m128* pSrc2 = (__m128*) pArray2;
	__m128* pDest = (__m128*) pResult;


	__m128 m0_5 = _mm_set_ps1(0.5f);        // m0_5[0, 1, 2, 3] = 0.5

	for ( int i = 0; i < nLoop; i++ )
	{
		m1 = _mm_mul_ps(*pSrc1, *pSrc1);        // m1 = *pSrc1 * *pSrc1
		m2 = _mm_mul_ps(*pSrc2, *pSrc2);        // m2 = *pSrc2 * *pSrc2
		m3 = _mm_add_ps(m1, m2);                // m3 = m1 + m2
		m4 = _mm_sqrt_ps(m3);                   // m4 = sqrt(m3)
		*pDest = _mm_add_ps(m4, m0_5);          // *pDest = m4 + 0.5

		pSrc1++;
		pSrc2++;
		pDest++;
	}
}
void ComputeArrayCPlusPlusAVX(
	float* pArray1,                   // [in] first source array
	float* pArray2,                   // [in] second source array
	float* pResult,                   // [out] result array
	int nSize)                        // [in] size of all arrays
{
	int nLoop = nSize/ 8;

	__m256 m1, m2, m3, m4;

	__m256* pSrc1 = (__m256*) pArray1;
	__m256* pSrc2 = (__m256*) pArray2;
	__m256* pDest = (__m256*) pResult;

	__m256 m0_5 = _mm256_set1_ps(0.5f);        // m0_5[0, 1, 2, 3] = 0.5

	for ( int i = 0; i < nLoop; i++ )
	{
		m1 = _mm256_mul_ps(*pSrc1, *pSrc1);        // m1 = *pSrc1 * *pSrc1
		m2 = _mm256_mul_ps(*pSrc2, *pSrc2);        // m2 = *pSrc2 * *pSrc2
		m3 = _mm256_add_ps(m1, m2);                // m3 = m1 + m2
		m4 = _mm256_sqrt_ps(m3);                   // m4 = sqrt(m3)
		*pDest = _mm256_add_ps(m4, m0_5);          // *pDest = m4 + 0.5

		pSrc1++;
		pSrc2++;
		pDest++;
	}
}

void ComputeArrayCPlusPlus(
	float* pArray1,                   // [in] first source array
	float* pArray2,                   // [in] second source array
	float* pResult,                   // [out] result array
	int nSize)                        // [in] size of all arrays
{

	int i;

	float* pSource1 = pArray1;
	float* pSource2 = pArray2;
	float* pDest = pResult;

	for ( i = 0; i < nSize; i++ )
	{
		*pDest = (float)sqrt((*pSource1) * (*pSource1) + (*pSource2)
			* (*pSource2)) + 0.5f;

		pSource1++;
		pSource2++;
		pDest++;
	}
}
void StartCounter()
{
    LARGE_INTEGER li;
    if(!QueryPerformanceFrequency(&li))
	std::cout << "QueryPerformanceFrequency failed!\n";

    PCFreq = double(li.QuadPart)/1000.0;

    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;
}
double GetCounter()
{
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return double(li.QuadPart-CounterStart)/PCFreq;
}


