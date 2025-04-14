Date: 2025-03-14
Course: https://www.techinterviewhandbook.org/algorithms/study-cheatsheet/
Chapter: 
Lecture: 
Topic: #coding #interview

## General recommendation
1. Validate input first or ask if input is valid (saves time)
2. Write pure functions as much as possible
3. "If you are cutting corners in your code, state that out loud to your interviewer and say what you would do in a non-interview setting"
4. As a last resort, just try each data structure and see which one works

## 1. Array
1. Arrays hold values of the *same type* at *contiguous memory locations*.
2. slicing and concatenating arrays would take O(n) time. Use start and end indices to demarcate a subarray/range where possible.
3. `subsequence` vs `subarray`

### [TwoSums](https://leetcode.com/problems/two-sum/) 
1. Good use case for `hashmap` i.e., `dictionary`. Gives `O(n)` complexity
### [3Sum](https://leetcode.com/problems/3sum/description/)
1. Sort, fix 1st number than two pointer. See [solution](https://youtu.be/IIxoo93bmPQ)
### [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)
1. Good example: identify *repeated* computation. Do NOT have to calculate those partial products every time!
	<details>
	 <summary>Click to see hint</summary>
	 * Calculate running prefix/suffix products for each element, then take the product of those two products.
	</details>
### [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/)
1. Very good [visualization](https://youtu.be/ioFPBdChabY): identify *repeated* computation. 

### [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/description/)
1. Example of *Kadane's* algorithm. Here is an [explanation](https://youtu.be/86CQq3pKSUw) on maximum sum
2. Idea is 
	1. to see the impact of existing sum/product on current element, i.e., compare `current_sum` with `nums[i]` and reset if negative impact from `current_sum`
## 2. String
### data structures
1. Tries: 
	1. stores a *set* of strings $S$. Prefix/suffix trie and compressed trie
		1. For standard trie, NO word in $S$ should be the prefix of another. Can be easily fixed by adding a boolean field $isWord$.
		2. Worst case space complexity $O(n)$ where $n$ is number of letters. This happens when words in $S$ has no common prefixes.
	2. See this [video](https://youtu.be/tUBFINxzlLg) and two following it for high-level; this [video](https://youtu.be/-urNrIAQnNo) for in-depth; this [video](https://youtu.be/oobqoCJlHA0) for implementation
2. Algo
	1. [Rabin-Karp](https://youtu.be/qQ8vS2btsxI)for pattern searching with a rolling hash
		1. compare pattern and substring char by char ONLY when their hash values match
		2. strong *hash function* leads to less value collision
	2. KMP for efficient searching of pattern
		1. Best [video](https://youtu.be/af1oqpnH1vA) to understand why it works
			1. Big idea is to AVOID back track on the main string (therefore $O(n)$), only move in the LPS ($O(m)$).
		2. [Video](https://youtu.be/GTJr8OvyEVQ) for KMP in action
		3. implementation?