# CUDA Shi Tomasi
Contents
--------

* CUDA Shi Tomasi Implementation
* CUDA Shi Tomasi Approximate Implementation
* Analysis

Description
-----------

The following is a parallelized implementation of Shi Tomasi. Shi Tomasi is an algorithm that is able to extract corners from an image. The algorithm begins by obtaining a horizontal and vertical gradient from the original image. It will then compute eigen values from the image. The eigen values will then be sorted and the largest values will be selected as features. Due to the having to work with large matrices, this would benefit highly from CUDA. The project also includes a performance analysis and scripts to obtain run times from the programs. 

Analysis Screenshots
--------------------

![Approximate Implementation Run Times](/ANALYSIS/approximation_runtimes.png)

![Parallel Run Times](/ANALYSIS/parallel_runtimes.png)

![Approximate Implementation Speedup](/ANALYSIS/approximation_speedup.png)

![Parallel Run speedup](/ANALYSIS/parallel_speedup.png)

