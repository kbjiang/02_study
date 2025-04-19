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
1. *Sort it!*
2. Then fix 1st number than two pointer. See [solution](https://youtu.be/IIxoo93bmPQ)
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
	3. When criteria (such as 'max/min') and grouped (such as 'subarray') are involved, try `sliding window`, i.e. `two pointers`.
		1. see this [section](https://youtu.be/MK-NZ4hN7rs?t=1030)
### [Longest Substring without repeating characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
1. The big idea is to iterate and ignore anything goes before the last time current character repeated 
2. Good [vid](https://youtu.be/pY2dYa1m2VM) with very nice use of hash. Use *two pointers*!

### [Encode and Decode Strings](https://neetcode.io/problems/string-encode-and-decode)
1. Key is to include string lengths in the encoded string.

## 3. Search
### [Binary search](https://leetcode.com/problems/binary-search/description/)
1. Use indices for searching problem!
	1. If the target is close to one end of array, the `mid` will first approach the target from one side then the other.
2. pay attention to edge cases when close to solution
3. needs `if arr[mid] == target` as the base case
	1. this is not needed if we are changing the value itself, instead of indices. See [[coding interview#[Kth smallest element in a sorted matrix](https //leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/description/)]]

### [Kth smallest element in a sorted matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/description/) using Binary search
1. Needs to calculate the number of elements smaller than `mid`. 
	1. See [Ref2](https://www.cnblogs.com/firecode7/p/16120445.html) for a good explanation; [Ref1](https://leetcode-solution-leetcode-pp.gitbook.io/leetcode-solution/medium/378.kth-smallest-element-in-a-sorted-matrix) for Python implementation
2. Here we are changing the value itself `1` every step, e.g., `right = mid - 1`
	1. `-1` is important otherwise infinite loop when `left==right`
	2.  Do NOT need `if counts == k` as there could be multiple `mid` fit the bill
	3. `right` will eventually ends up too small by 1, that's why we need `left` to pass `right` before return `left`.
		1. As opposed to [Sqrt(x)](https://leetcode.com/problems/sqrtx/) where `right` is returned because we are looking for round down
	4. See [Ref1](https://leetcode-solution-leetcode-pp.gitbook.io/leetcode-solution/medium/378.kth-smallest-element-in-a-sorted-matrix) for Python solution

## 4. Matrix
### [Set Matrix Zeroes](https://youtu.be/T41rL0L3Pnw)
1. idea is to use 1st row and 1st column as indicator
2. for $O(1)$ solution, the entry at $(0, 0)$ needs an extra memory because it cannot be the indicator for both 1st row and 1st column

### [Spiral Matrix](https://leetcode.com/problems/spiral-matrix/description/)
1. Very nice [solution](https://leetcode.com/problems/spiral-matrix/solutions/3502927/c-java-python-image-explanation-recursion-and-single-for-loop/) 
	1. recursion on each edge, i.e., from $m \times n$ to $n \times (m-1)$
	2. use `dr, dc` to control change of direction
	3. starting point is $(0, -1)$

### [Valid Sudoku](https://leetcode.com/problems/valid-sudoku/description/) (coding trick)
1. Nice [solution](https://youtu.be/TjFXEUCMqI8) with use of `defaultdict(set)`, it's easier than regular `dict` coz it provides a default value for keys don't exist yet.

## 5. Binary Search Trees
### data structures
1. `node`, `keye`, `root`, `child`, `depth` 
2. Traversal 

### [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)
1. This [time stamp]](https://youtu.be/K81C31ytOZE?t=281) shows how to deal with global/nonlocal variable
2. This [video](https://youtu.be/81lu4qO9snY) has very good visualization of recursion

### [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/description/)
1. BFS and queue
	1. `while queue` is the condition to terminate loop when queue becomes empty
2. Good [explanation](https://youtu.be/KFkjJ7pjWVw)