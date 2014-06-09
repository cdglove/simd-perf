// simd-mult.cpp
// 
// cl.exe /EHsc /Ox simd-mult.cpp
// g++ -std=c++11 -O3 simd-mult.cpp
//
// or
// 
// cl.exe /EHsc /Ox /arch:AVX2 simd-copy.cpp
// g++ -std=c++11 -O3 -march=core-avx2 -mtune=core-avx2 -mavx2 

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <immintrin.h>
#include <cassert>
#include "cgutil/program_options.h"
#include "cgutil/timer.h"

#ifndef SUPPORT_AVX
#  define SUPPORT_AVX 1
#endif

template<typename T>
T* align(T* p, std::size_t aligned_to)
{
	std::size_t address = reinterpret_cast<std::size_t>(p);
	while(address % 256 != aligned_to)
		++address;
	return reinterpret_cast<T*>(address);
}

// ----------------------------------------------------------------------------
//
std::size_t kDefaultNumFloats = 16 * 1024;
std::size_t kDefaultTotalFloats = 65636 * kDefaultNumFloats;
std::size_t gNumFloats = kDefaultNumFloats;
std::size_t gTotalFloats = kDefaultTotalFloats;
float gCheckValue = 1.f;
bool gHasAvx = true;
bool gHtmlOut = true;

void NiaveMult(float* d, float const* a, float const* b)
{
	for(int i = 0; i < gNumFloats; ++i)
	{
		*d++ = *a++ * *b++;
	}
}

void UnalignedSseMult(float* d, float const* a, float const* b)
{
	for(int i = 0; i < gNumFloats; i += 4)
	{
		__m128 v1 = _mm_loadu_ps(&a[i]);
		__m128 v2 = _mm_loadu_ps(&b[i]);
		__m128 r = _mm_mul_ps(v1, v2);
		_mm_storeu_ps(&d[i], r);
	}
}

void AlignedSseMult(float* d, float const* a, float const* b)
{
	for(int i = 0; i < gNumFloats; i += 4)
	{
		__m128 v1 = _mm_load_ps(&a[i]);
		__m128 v2 = _mm_load_ps(&b[i]);
		__m128 r = _mm_mul_ps(v1, v2);
		_mm_store_ps(&d[i], r);
	}
}

void AlignedSseNonTemporalMult(float* d, float const* a, float const* b)
{
	for(int i = 0; i < gNumFloats; i += 4)
	{
		__m128 v1 = _mm_load_ps(&a[i]);
		__m128 v2 = _mm_load_ps(&b[i]);
		__m128 r = _mm_mul_ps(v1, v2);
		_mm_stream_ps(&d[i], r);
	}
}

#if SUPPORT_AVX
void UnalignedAvxMult(float* d, float const* a, float const* b)
{
	for(int i = 0; i < gNumFloats; i += 8)
	{
		__m256 v1 = _mm256_loadu_ps(&a[i]);
		__m256 v2 = _mm256_loadu_ps(&b[i]);
		__m256 r = _mm256_mul_ps(v1, v2);
		_mm256_storeu_ps(&d[i], r);
	}
}

void AlignedAvxMult(float* d, float const* a, float const* b)
{
	for(int i = 0; i < gNumFloats; i += 8)
	{
		__m256 v1 = _mm256_load_ps(&a[i]);
		__m256 v2 = _mm256_load_ps(&b[i]);
		__m256 r = _mm256_mul_ps(v1, v2);
		_mm256_store_ps(&d[i], r);
	}
}

void AlignedAvxNonTemporalMult(float* d, float const* a, float const* b)
{
	for(int i = 0; i < gNumFloats; i += 8)
	{
		__m256 v1 = _mm256_load_ps(&a[i]);
		__m256 v2 = _mm256_load_ps(&b[i]);
		__m256 r = _mm256_mul_ps(v1, v2);
		_mm256_stream_ps(&d[i], r);
	}
}
#endif

void NullMult(float*, float const*, float const*)
{}

// ----------------------------------------------------------------------------
//
template<void(*f)(float*, float const*, float const*)>
void Run(char const* name, std::size_t alignment, float* d, float const* a, float const* b)
{
	d = align(d, alignment);
	a = align(a, alignment);
	b = align(b, alignment);
	std::fill(d, d + gNumFloats, 0.f);

	cgutil::timer t;
	for(std::size_t i = 0; i < gTotalFloats; i += gNumFloats)
	{
		f(d, a, b);
	}
	float time = t.elapsed();

	for(std::size_t i = 0; i < gNumFloats; ++i)
	{
		if(d[i] != a[i] * b[i])
		{
			std::cerr << "Error in " << name << " " << d[i] << " != " << a[i] * b[i] << std::endl;
			std::exit(1);
		}
	}

	std::cerr << name 
			  << " (" << alignment << ") took " 
			  << time << " seconds." 
			  << std::endl
	;

	std::cout << "," << time;
}

template<>
void Run<NullMult>(char const*,  std::size_t, float*, float const*, float const*)
{
	std::cout << "," << 0;
}


// ----------------------------------------------------------------------------
//
void print_usage()
{
	std::cerr << "Usage:\n" 
			  << "simd-mult [options]\n"
			  << "num-floats=<number of float in memory>    default (" << kDefaultNumFloats << ")\n"
			  << "total-floats=<number of floats total>     default (" << kDefaultTotalFloats << ")\n"
			  << "check-value=<any value to check against>  default (" << gCheckValue << ")\n"
			  << "enable-avx=<true/false>                   default (" << std::boolalpha << gHasAvx << ")\n"
			  << "report-html=<true/false>                  default (" << std::boolalpha << gHtmlOut << ")\n"
			  << std::endl;

}

// ----------------------------------------------------------------------------
//
int main(int argc, char** argv)
{
	cgutil::program_options opts;
	opts.add("num-floats", gNumFloats);
	opts.add("num-floats", gNumFloats);
	opts.add("total-floats", gTotalFloats);
	opts.add("check-value", gCheckValue);
	opts.add("enable-avx", gHasAvx);
	
	try
	{
		opts.parse(argc, argv);
	}
	catch(std::runtime_error e)
	{
		std::cerr << e.what() << std::endl;
		print_usage();
		std::exit(1);
	}

	if(gTotalFloats < gNumFloats)
	{
		std::cerr << "total-floats must be greater than num-floats" << std::endl;
		print_usage();
		return 0;
	}

	if(gHtmlOut)
	{
		std::cout <<
		   "<html>\n"
		   "  <head>\n"
		   "    <script type=\"text/javascript\" src=\"https://www.google.com/jsapi\"></script>\n"
		   "    <script type=\"text/javascript\">\n"
		   "      google.load(\"visualization\", \"1\", {packages:[\"corechart\"]});\n"
		   "      google.setOnLoadCallback(drawChart);\n"
		   "      function drawChart() {\n"
		   "        var data = google.visualization.arrayToDataTable([\n"
		;
	}

	// Allocate 64 megs worth of floats;
	std::vector<float> source(gNumFloats + 0x1000, gCheckValue);
	std::vector<float> dest(gNumFloats + 0x100, 0.f);

	std::cout << "[\'Alignment\',\'for-loop\',\'Unaligned Sse\',\'Unaligned Avx\',\'Aligned Sse\',\'Aligned Sse Stream\',\'Aligned Avx\',\'Aligned Avx Stream\'";
	for(std::size_t alignment = 4; alignment <= 64; ++alignment)
	{
		std::cout << "],\n" << "[" << alignment;

		Run<NiaveMult>("for-loop", alignment, dest.data(), source.data(), source.data() + 256);
		Run<UnalignedSseMult>("Unaligned Sse", alignment, dest.data(), source.data(), source.data() + 256);

	#if SUPPORT_AVX
		if(gHasAvx)
		{
			Run<UnalignedAvxMult>("Unaligned Avx", alignment, dest.data(), source.data(), source.data() + 256);
		}
		else
	#endif
		{
			Run<NullMult>("Unaligned Avx", alignment, source.data(), source.data(), source.data() + 256);
		}

		if(alignment % 16 == 0)
		{
			Run<AlignedSseMult>("Aligned Sse", alignment, dest.data(), source.data(), source.data() + 256);
			Run<AlignedSseNonTemporalMult>("Aligned Sse Stream", alignment, dest.data(), source.data(), source.data() + 256);
		}
		else
		{
			Run<NullMult>("Aligned Sse", alignment, source.data(), source.data(), source.data() + 256);
			Run<NullMult>("Aligned Sse Stream", alignment, source.data(), source.data(), source.data() + 256);
		}
	#if SUPPORT_AVX
		if(alignment % 32 == 0 && gHasAvx)
		{
			Run<AlignedAvxMult>("Aligned Avx", alignment, dest.data(), source.data(), source.data() + 256);
			Run<AlignedAvxNonTemporalMult>("Aligned Avx Stream", alignment, dest.data(), source.data(), source.data() + 256);
		}
		else
	#endif
		{
			Run<NullMult>("Aligned Avx", alignment, source.data(), source.data(), source.data() + 256);
			Run<NullMult>("Aligned Avx Stream", alignment, source.data(), source.data(), source.data() + 256);
		}       
	}

	std::cout << "]" << std::endl;

	if(gHtmlOut)
	{
		std::cout <<
			"        ]);\n"
			"        var options = {\n"
			"          title: 'Alignment vs. Run Time'\n"
			"        };\n"
			"        var chart = new google.visualization.LineChart(document.getElementById('chart_div'));\n"
			"        chart.draw(data, options);\n"
			"      }\n"
			"    </script>\n"
			"  </head>\n"
			"  <body>\n"
			"    <div id=\"chart_div\" style=\"width: 900px; height: 500px;\"></div>\n"
			"  </body>\n"
			"</html>\n"
		;
	}


	return 0;
}