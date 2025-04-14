## Divide and Conquer, Sorting and Searching, and Randomized Algorithms
### Module 1
1. Asymptotic notation in seven words: *suppress constant factors and lower-order terms*. 
	1. constant factors: system dependent
	2. lower-order: irrelevant for large inputs
2. Big-O notation
	1. By $T(n)=O(f(n))$ we mean $f(n)$ is the upper bound, not necessarily tight, of $T(n)$. Think of $T(n)$ as the number of ops and $f(n)$ is usually something like linear, quadratic or logarithmic.
		1. Don't get confused by the mathematical definition. It's just about finding the right bounds to assess $T(n)$ in a asymptotic way.
3. Divide & Conquer
	1. usually the combine step is most challenging

### Module 2
1. Master equation
	1. rate of proliferation vs rate of shrinking
	2. Intuition of three cases
		1. equal
		2. root dominant
		3. leaf dominant

### Module 3
1. Quicksort
	1. The power horse is the `partition` subroutine which 
		1. move all elements $<p$ to the left of pivot $p$ and all $>p$ to its right; sets up for recursive calls
		2. consequently, $p$ is at the right position now and is done
	2. in-place
	3. the analysis of running time of randomized Quicksort is beautiful. https://youtu.be/sToWtKSYlMw
		1. Identify the target random variable
		2. Decompose into smaller r.v., i.e., indicators
		3. The spirit again is to divide a complicated problem into smaller, manageable parts
### Module 4
1. random is good proxy of best-case scenario
		