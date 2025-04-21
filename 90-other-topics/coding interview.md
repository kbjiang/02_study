Date: 2025-03-14
Course: https://www.techinterviewhandbook.org/algorithms/study-cheatsheet/
Chapter: 
Lecture: 
Topic: #coding #interview

## My understanding
1. Be familiar with data structures and their regular operations
	1. E.g., Binary trees and its traversal
2. Be familiar with basic coding, such as recursion
3. The trick or *brain teasing* part should not be the focus.
	1. The focus should be problem formulation and get familiar with the data structure
	2. E.g., the sum of linked list. They actually hinted the trick part
4. The complexity requirements: always look for less repeated computation
	2. E.g., the two pointer problems
## General recommendation
1. Validate input first or ask if input is valid (saves time)
2. Write pure functions as much as possible
3. "If you are cutting corners in your code, state that out loud to your interviewer and say what you would do in a non-interview setting"
4. As a last resort, just try each data structure and see which one works

## - Array
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
## - String
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

## - Sorting and Search
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

## - Matrix
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

## - Binary Search Trees
### data structures
1. Traversal. See how the append is done.
	```python
	class TreeNode:
	    def __init__(self, key):
	        self.key = key
	        self.left = None
	        self.right = None

	def in_order_traversal(root):
		result = []		
		def traverse(node):		
			if node is not None:		
				traverse(node.left)  # Visit left subtree		
				result.append(node.key)  # Visit current node		
				traverse(node.right)  # Visit right subtree		
		
		traverse(root)
		return result
	```
2. For level wise question, use `queue`.
3. Binary Tree vs Binary Search Tree
4. A lot of recursive calls
	2. usually use `if not root`, i.e., None node, as base case
	3. usually something like `f(root.val) & g(root.left) & g(root.right)` where `f` is the regular operation and $g$ is the recursive call

### [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)
1. This [time stamp](https://youtu.be/K81C31ytOZE?t=281) shows how to deal with global/nonlocal variable, i.e., use `self.res`
2. This [video](https://youtu.be/81lu4qO9snY) has very good visualization of recursion

### [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/description/)
1. Breadth First Search (BFS) and *queue*
	1. `while queue` is the condition to terminate loop when queue becomes empty
	2. `qLen = len(q)` is important coz the length of `q` will change during loop
		1. `queue` is 1st in 1st out; add node's children before removing it
		2. The space complexity is `O(n)` coz last level could be `n/2`
2. Good [explanation](https://youtu.be/KFkjJ7pjWVw) and Python [implementation](https://youtu.be/6ZnyEApgFYg?t=393)
### [Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)
1. [Solution](https://www.jiakaobo.com/leetcode/236.%20Lowest%20Common%20Ancestor%20of%20a%20Binary%20Tree.html)
2. The challenge for me is to know *what to return* at each recursion
	1. Idea is to return upwards till we find a root with both `left` and `right` not null.
		1. Therefore `if left and right: return root` and `return left if left else right`
		2. Tricky when one target node is the ancestor `q` of the other `p`. 
			1. No need to explore more, because if `p` is not a child of `q`, the peer of `q` will return non-null value, so the ancestor will be a parent of `q`
			2. see example 2 with `p=5, q=4`.
	2. `if not root or root == p or root == q: return root`
		2. This is base case.

### [572. Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/description/)
1. [Solution](https://youtu.be/E36O5SWp-LE)

### [98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
1. Brutal force will take $O(n^2)$
2. [Solution](https://youtu.be/s6ATEkipzow) uses boundaries from root and is $O(n)$.
	1. to see how the boundaries work, just think the limits of `root.right` and `root.right.left`, the `root.val` naturally becomes the loweer limit of `root.right.left` 


## - Graph
### Data structure
1. [Three representations](https://medium.com/basecs/from-theory-to-practice-representing-graphs-cfd782c5be38)
	1. edge list, adjacency matrix, adjacency list
	2. directed vs undirected
	3. Hash table of hash table?
2. Traversal ([BFS](https://medium.com/basecs/going-broad-in-a-graph-bfs-traversal-959bd1a09255) and [DFS](https://medium.com/basecs/deep-dive-through-a-graph-dfs-traversal-8177df5d0f13))
	1. Data structure
		1. BFS with queue, needs `popleft` and `append`
		2. DFS with stack, no popping?
		3. Implementation [here](https://www.techinterviewhandbook.org/algorithms/graph/)
			1. Graphs are usually matrix?
	2. For a give node, 
		2. BFS iterates over all of its adjacency linked list, i.e., neighbors at same level, thus breadth
		3. DFS will recurse on the 1st element and put rest of the linked list "on hold" until backtracking, therefore prioritize depth
	3. BFS tells us the *shortest* path from node $a$ to node $b$; DFS tells us if a path from $a$ to $b$ even exists
	4. BFS iterative, DFS recursive

### 542. [01 Matrix](https://leetcode.com/problems/01-matrix/)
1. first find *all possible starting points*, then run BFS. See [solution](https://youtu.be/YTnYte6U61w)
	1. This can be thought of as all starting points are 1 step from a single super starting point.

### 133. [Clone Graph](https://leetcode.com/problems/clone-graph/)
1. DFS and update values dynamically. [solution](https://youtu.be/mQeF6bN8hMk)

Tow-D Maze