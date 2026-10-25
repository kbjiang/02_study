#coding #interview

## My understanding
1. Be familiar with problem formulation, data structures and their regular operations
	1. E.g., Binary trees and its traversal
	2. The trick or *brain teasing* part should not be the focus.
		1. E.g., the sum of linked list. They actually hinted the trick part
		
2. Be familiar with basic coding, such as recursion
3. The complexity requirements: always look for less repeated computation
	1. E.g., the `two sum` problem
## General recommendation
1. Read the problem carefully, each sentence can be useful.
	1. E.g., in `two sums`, "each input would have **_exactly_ one solution**" is important.
2. Validate input first or ask if input is valid (saves time)
3. Write pure functions as much as possible
4. "If you are cutting corners in your code, state that out loud to your interviewer and say what you would do in a non-interview setting"
5. As a last resort, just try each data structure and see which one works

## Methodology
1. Bottom line
	1. prioritize core data structures + pattern fluency way over memorizing clever tricks
	2. therefore no need to do a lot of LC problems
2. Follow 5-week plan (organized by topics) in https://www.techinterviewhandbook.org/best-practice-questions/
	1. When stuck, find solution at https://neetcode.io/practice/practice/allNC
	2. At the end I should redo problems without knowing its topics
	3. Cross check https://neetcode.io/roadmap
3. A better version is [Grind75](https://www.techinterviewhandbook.org/grind75/) where the list is updated for different time lines.
4. Exercise notebook: [coding-interview.ipynb](C:\Users\kejia\Documents\GitHub\02_study\90-other-topics\interviews\)

---
## Array
1. Arrays hold values of the *same type* at *contiguous memory locations*.
2. Use start and end indices to demarcate a subarray/range wherever possible.
3. `subsequence` vs `subarray`
4. Time complexity

| Operation             | Big-O     | Note                                                                                                 |
| --------------------- | --------- | ---------------------------------------------------------------------------------------------------- |
| Access                | O(1)      |                                                                                                      |
| Search                | O(n)      |                                                                                                      |
| Search (sorted array) | O(log(n)) |                                                                                                      |
| Insert                | O(n)      | Insertion would require shifting all the subsequent elements to the right by one and that takes O(n) |
| Insert (at the end)   | O(1)      | Special case of insertion where no other element needs to be shifted                                 |
| Remove                | O(n)      | Removal would require shifting all the subsequent elements to the left by one and that takes O(n)    |
| Remove (at the end)   | O(1)      | Special case of removal where no other element needs to be shifted                                   |
#### [TwoSums](https://leetcode.com/problems/two-sum/) 
1. Good use case for `hashmap` i.e., `dictionary`. Gives `O(n)` complexity
2. Why is `hashmap` lookup only `O(1)`?
#### [3Sum](https://leetcode.com/problems/3sum/description/)
1. Sort first. Then fix 1st number and run two pointers. See [solution](https://youtu.be/jzZsG8n2R9A)
#### [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)
1. Calculate running prefix/suffix products for each element, then take the product of those two products.
2. Can be done with space `O(1)` without calculating prefix/suffix separately.
#### [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/)
1. Very good [visualization](https://youtu.be/ioFPBdChabY): identify *repeated* computation. 

#### [Maximum Product Subarray](https://leetcode.com/problems/maximum-product-subarray/description/)
1. Example of *Kadane's* algorithm. Here is an [explanation](https://youtu.be/86CQq3pKSUw) on maximum sum
2. Idea is 
	1. to see the impact of existing sum/product on current element, i.e., compare `current_sum` with `nums[i]` and reset if negative impact from `current_sum`
#### Sliding window (important technique)
1. Interview: you are given an array of 1's and 0's. You are also given a number k. What is the length of the longest continuous subsequence of 1's you can pick from the given array? You can substitute at most k 0's with 1's. Write a script in python.
	1. Example: input [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1], k=2;	output: 8
	2. The mental picture is looking for a sliding window with no more than k zeros.

---
## Hash table
#### Learning
 1. Idea
	 1. replace search $O(n)$ with array indexing $O(1)$.
		 1. Get the key of the target object, and go directly to the corresponding *bucket*, without scanning.
 2. Space-time trade-off
	 1. if use unique integer keys for each object, there will be a lot of wasted space to store the set of all possible keys $U$. 
		 1. also called "Direct-address table"
 3. Hash function
	 1. Instead of $U$, design $h$ to shrink array size to $\Theta(n)$.
		 1. Therefore collisions
 4. References
	 1. [Hash tables: Samuel's tutorial](https://youtu.be/r1XZGP5ppqQ?list=PL9t0xVFP90GBPScODccB2_PpouUUZ4j8C)

---
## Recursion
### Learning
1. Parts of every recursion
	1. Base case, i.e., when to stop
		1. *This also includes when condition is met*, e.g., search target found and stop.
		2. E.g., LeetCode 22 base case is `len(s)==2*n`. *Maximum* size as base case.
	2. Recursively work towards base case
		1. Dealing with intermediate results is important. See discussion ([[#^1361ef]] ) below.
2. Three types of application of recursion
	1. Memorization
		1. Use universial cache to record intermediate results
	2. Divide-and-conquer
		1. applied when needs results from all sub-problems
	3. Backtracking
		1. applied when needs to return all qualified answers
		2. *trickier because there is a retraction component*.
			1. *Attempt* every possible next step, *retract* as soon as it becomes invalid or reaches base case.
			2. E.g., LeetCode 22, valid expression with $n$ parentheses. Recursion prevents going further down invalid paths.
3. Intermediate results ^1361ef
	1. Memorization: when there's a lot repetitions. E.g., Fibonacci numbers. # 509
	2. Divide-and-conquer: combine the results of each subproblem
	3. *Backtracking*: look for *all* valid results
		1. Since we need to come back to the original state, needs to make sure the *progress* is well preserved
		2. Sometimes uses `list` with `.pop()` to store progress
	4. ==DFS: backtracking + mark visited?==
		1. relation with backtracking? E.g, maze problem
	5. How `return` works?
4. Why Recursion Works
	1. In a recursive algorithm, the computer "remembers" every previous state of the problem. This information is "held" by the computer on the **"activation stack"** (i.e., inside of each functions workspace). Every function has its own workspace *PER CALL* of the function.
5. Ref
	1. [硅谷工程师十五分钟带你深入理解](https://youtu.be/AqGagBmFXgw)
### Coding
#### [509. Fibonacci]()
1. w/ and w/o memorization ![[Pasted image 20261016171100.png|500]]

#### [22. Generate parenthesis](https://leetcode.com/problems/generate-parentheses/)
1. base case happens at max length, i.e., only valid full length strings make it to the result list.
2. recursion conditions, `n_left < n` select valid paths
```python
def generateParenthesis(self, n: int) -> list[str]:
	res = []
	def backtrack(n_left, n_right, s) -> None:
		# n_left: num of left parenthesis
		# n_right: num of right parenthesis
		
		# base case
		if len(s) == n*2:
			res.append(s)
			# return 
		
		# either `if+return` or `if+else``
		else:
			# attempt to add left parenthesis
			if n_left < n:
				backtrack(n_left + 1, n_right, s+"(")
			# attempt to add right parenthesis
			if n_right < n_left: # num of rights never exceeds num of lefts
				backtrack(n_left, n_right+1, s+")")

	backtrack(0, 0, "")
	return res
```
#### [77. Combinations](https://leetcode.com/problems/combinations/)
1. the explicit backtrack of `lst`
2. use of `lst.copy()` when appending
```python
def combine(self, n: int, k: int) -> list[list[int]]:
	res = []
	def backtrack(lst, cans):
		# base case
		if len(lst) == k:		
		# .copy to not affect items in `res`
		res.append(lst.copy())
		return
		
		# attempt possible actions
		for i, can in enumerate(cans):
			lst = lst + [can]
			backtrack(lst, cans[i+1:])
			# to backtrack
			lst.pop()

	backtrack([], list(range(1, n+1)))
	return res
```
## String
### Learning
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
### Coding
#### [Longest Substring without repeating characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
1. The big idea is to iterate and ignore anything goes before the last time current character repeated 
2. Good [vid](https://youtu.be/pY2dYa1m2VM) with very nice use of hash. Use *two pointers*!

#### [Encode and Decode Strings](https://neetcode.io/problems/string-encode-and-decode)
1. Key is to include string lengths in the encoded string.

## Sorting and Search
#### [Binary search](https://leetcode.com/problems/binary-search/description/)
1. Use indices/pointers for searching problem!
	1. If the target is close to one end of array, the `mid` will first approach the target from one side then the other.
2. pay attention to edge cases when close to solution
	1. the mid points and the end condition `l < r` or `l <= r` or `l < r-1`
	2. return `l` or `r`?
3. needs `if arr[mid] == target` as the base case
	1. this is not needed if we are changing the value itself, instead of indices. See [[coding-interview#[Kth smallest element in a sorted matrix](https //leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/description/)]]
#### [Kth smallest element in a sorted matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/description/) using Binary search
1. Needs to calculate the number of elements smaller than `mid`. 
	1. See [Ref2](https://www.cnblogs.com/firecode7/p/16120445.html) for a good explanation; [Ref1](https://leetcode-solution-leetcode-pp.gitbook.io/leetcode-solution/medium/378.kth-smallest-element-in-a-sorted-matrix) for Python implementation
2. Here we are changing the value itself `1` every step, e.g., `right = mid - 1`
	1. `-1` is important otherwise infinite loop when `left==right`
	2.  Do NOT need `if counts == k` as there could be multiple `mid` fit the bill
	3. `right` will eventually ends up too small by 1, that's why we need `left` to pass `right` before return `left`.
		1. As opposed to [Sqrt(x)](https://leetcode.com/problems/sqrtx/) where `right` is returned because we are looking for round down
	4. See [Ref1](https://leetcode-solution-leetcode-pp.gitbook.io/leetcode-solution/medium/378.kth-smallest-element-in-a-sorted-matrix) for Python solution

#### [26. Remove Duplicates from Sorted Array](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)
1. Use left pointer $l$ to make sure `a[:l]` are the unique elements ran into so far. This is crucial for in-place.
## Matrix
### Coding
#### [Set Matrix Zeroes](https://youtu.be/T41rL0L3Pnw)
1. idea is to use 1st row and 1st column as indicator
2. for $O(1)$ solution, the entry at $(0, 0)$ needs an extra memory because it cannot be the indicator for both 1st row and 1st column

#### [Spiral Matrix](https://leetcode.com/problems/spiral-matrix/description/)
1. Very nice [solution](https://leetcode.com/problems/spiral-matrix/solutions/3502927/c-java-python-image-explanation-recursion-and-single-for-loop/) 
	1. recursion on each edge, i.e., from $m \times n$ to $n \times (m-1)$
	2. use `dr, dc` to control change of direction
	3. starting point is $(0, -1)$
#### [Valid Sudoku](https://leetcode.com/problems/valid-sudoku/description/) (coding trick)
1. Nice [solution](https://youtu.be/TjFXEUCMqI8) with use of `defaultdict(set)`, it's easier than regular `dict` coz it provides a default value for keys don't exist yet.


## Linked list
### Data structure
1. basic form: each node contains `data` and `reference` to next node
2. time complexity:
	1. `O(n)` for access and search
	2. `O(1)` for insertion and deletion
3. Do NOT require contiguous memory allocation, as opposed to arrays
	1. arrays are static data structures, linked lists are dynamic
4. Singly and Doubly linked
### Tricks
1. Think of `node` as pointing to the *entire linked list starts at `node`*, not just a single node.
	1. E.g., [[#[206. Reverse Linked List](https //leetcode.com/problems/reverse-linked-list/)]] for a linked list `1->2->3->4`, the recursion is `reverse(1) = connect(1, reverse(2)) = connect(1, connect(2, reverse(3)))`, not `reverse(1) = connect(1, reverse(2, 3, 4))`
	2. E.g., [21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/). When only `list1` is left non-empty, one  can just do `cur.next = list1` to connect the whole list
2. Notice when *it's just pointing vs when it's changing* values!
	1. [[#[21. Merge Two Sorted Lists](https //leetcode.com/problems/merge-two-sorted-lists/)]]

### Techniques
1. Adding a sentinel/dummy node at the head and/or tail might help to handle many edge cases where operations have to be performed at the head or the tail.
	1. E.g., [[#[21. Merge Two Sorted Lists](https //leetcode.com/problems/merge-two-sorted-lists/)]]
2. two pointers
	1. one is $k$ ahead the other
	2. one is faster than the other
### [206. Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/)
1. use of two pointers `prev, curr`.
	```python
	    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
	        prev, curr = None, head
	        while curr:
	            nxt = curr.next
	            curr.next = prev
	            prev = curr
	            curr = nxt
	        return prev
	```
	1. This actually changes the physical data and `head` will be affected! 
2. For recursive, see this [solution](https://youtu.be/W-EfGB0E_ao). 
	1. It's not necessary at all. The recursive basically does the same thing as the iterative solution.
### [21. Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/)
1. Use of *pointers*!
	1. `tail` pointer to construct merged list
	2. extra `head` pointer to stay at the head of the whole list--so that we can return the whole list.
	3. Do `list1 = list1.next` ==prior to== `tail.next = list2`, otherwise `list1.next` will be overwritten by `list2`
2. compare `tail = tail.next` and `tail.next = tail.next.next`. The former is just moving the pointer, *the latter changes the actual data* in memory.
```python
def mergeTwoLists(self, list1: ListNode | None, list2: ListNode | None) -> ListNode | None:
	# initialize two pointers
	# tail to update list, head for return list
	head = ListNode()
	tail = head  
	
	# update list is trivial
	while list1 and list2:
		if list1.val <= list2.val:
			tail.next = list1
			list1 = list1.next
		else:
			tail.next = list2
			list2 = list2.next
		tail = tail.next
		
	# remainders
	if list1:
		tail.next = list1
	if list2:
		tail.next = list2
	return head.next
```
### [143. Reorder List](https://leetcode.com/problems/reorder-list/) $\star$
1. ==This problem subsumes previous problems.==
2. [Solution](https://youtu.be/S5bfdUTrKLM)
	1. Find the middle point
	2. Reverse the 2nd half of the list *in place*. 
	3. merge
3. when merging, be sure to move `first` pointer before doing `first.next=second`. 
```python
def reorderList(self, head: ListNode | None) -> None:
	"""
	Do not return anything, modify head in-place instead.
	"""
	
	# find middle, i.e., slow
	slow, fast = head, head.next
	while fast and fast.next:
		slow = slow.next
		fast = fast.next.next
	
	# reverse second half
	second = slow.next
	# to indicate the end of 1st half
	slow.next = None
	# the usual helper `prev` for reversing 2nd half
	prev = None
	
	while second:
		tmp = second.next
		second.next = prev
		prev = second
		second = tmp
	
	# merge two halves
	first, second = head, prev
	while second:
		tmp1, tmp2 = first.next, second.next
		first.next = second
		second.next = tmp1
		first, second = tmp1, tmp2
	
	return head
```

## Queue
### Learning
1. dequen from front, enqueue from back, FIFO
	1. dequeue only removes 1st element, which is different from removal
2. when not knowing the queue size, implement with linked list
### Coding
#### [232. Implement Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/) $\star$
1. The amortization is moving elements from stack 1 to stack 2 only when latter is empty. Moving is the expensive `O(n)` operation.
```python
def pop(self) -> int:
	if not self.s2:
		while self.s1:
			self.s2.append(self.s1.pop())
	return self.s2.pop()
```
#### [225. Implement Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues/description/)
1. Try to do with just one queue.

## Stack
### Learning
1. Use cases: recursion calls, undo operations, etc.
2. FILO
### Coding
#### [155. Min Stack](https://leetcode.com/problems/min-stack/)
1. I did it with a `min` flag; but I should have prioritized using a/intern extra stack!
#### [739. Daily Temperatures](https://leetcode.com/problems/daily-temperatures/)
1. See my solution online
#### [227 Basic Calculator II](https://leetcode.com/problems/basic-calculator-ii/)
1. Nice [solution](https://www.youtube.com/watch?v=iua49ZFeptY&t=521s)
	1. with details on how to *extract full numbers from string*. My online solution used `split()`.
	2. Take high priority calculation on the fly. At the end, just need to take the sum of the stack.
		1. E.g.,  `-num` is `-1*num` and do `stack.push(-num)` on the fly.

## Tree
### Learning
1. [Binary Search Trees: Samuel's tutorial](https://youtu.be/0woI8l0ZWmA)
	1. Nice animation on *tree traversals*
2. Time complexity is usually the height of the tree, i.e., `O(log(n))`
3. The main difference between a **binary tree** and a **binary search tree (BST)** is that a binary tree has **no specific ordering rules** for its nodes, whereas a binary search tree enforces a **strict sorted order constraint**
4. Traversal. See how the append is done.
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
5. For level wise question, use `queue`.
6. A lot of recursive calls
	1. Return values
		1. Top Down: base case return `None`. E.g., `inorder traversal` of trees; E.g., backtracking
		2. Bottom up: need value from subproblem, so return cannot be null.
	2. usually use `if not root`, i.e., None node, as base case
### Coding
#### [124. Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/)
1. Nice explanation in [tutorial](https://youtu.be/5jz66VkGMCY?list=PLV5qT67glKSErHD66rKTfqerMYz9OaTOs&t=1003)
	1. The recursion can ONLY have either left or right to avoid double visit of root. E.g., 3->15->5->15->20 in figure below
	2. To include contribution from both ends, at each step do `cur_sum = recurse(root.left) + recurse(root.right) + root.val`
	3. ![[Pasted image 20250519110557.png|800]]
#### [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/)
1. This [time stamp](https://youtu.be/K81C31ytOZE?t=281) shows how to deal with global/nonlocal variable, i.e., use `self.res`
2. This [video](https://youtu.be/81lu4qO9snY) has very good visualization of recursion
#### [Binary Tree Level Order Traversal](https://leetcode.com/problems/binary-tree-level-order-traversal/description/)
1. Breadth First Search (BFS) and *queue*
	1. `while queue` is the condition to terminate loop when queue becomes empty
	2. `qLen = len(q)` is important coz the length of `q` will change during loop
		1. `queue` is 1st in 1st out; add node's children before removing it
		2. The space complexity is `O(n)` coz last level could be `n/2`
2. Good [explanation](https://youtu.be/KFkjJ7pjWVw) and Python [implementation](https://youtu.be/6ZnyEApgFYg?t=393)
#### [236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)
1. The challenge for me is to know *what to return* at each recursion
	1. Idea is to return upwards till we find a root with both `left` and `right` not null.
		1. Therefore `if left and right: return root` and `return left if left else right`
		2. Tricky when one target node is the ancestor `q` of the other `p`. 
			1. No need to explore more, because if `p` is not a child of `q`, the peer of `q` will return non-null value, so the ancestor will be a parent of `q`
			2. see example 2 with `p=5, q=4`.
	2. `if not root or root == p or root == q: return root`
		2. This is base case.

#### [572. Subtree of Another Tree](https://leetcode.com/problems/subtree-of-another-tree/description/)
1. [Solution](https://youtu.be/E36O5SWp-LE)
#### [98. Validate Binary Search Tree](https://leetcode.com/problems/validate-binary-search-tree/)
1. Brutal force will take $O(n^2)$
2. [Solution](https://youtu.be/s6ATEkipzow) uses boundaries from root and is $O(n)$.
	1. to see how the boundaries work, just think the limits of `root.right` and `root.right.left`, the `root.val` naturally becomes the lower limit of `root.right.left` 
## Graph
### Learning
1. Representations
	1. edge list, adjacency matrix, adjacency list
	2. directed vs undirected
	3. ![[Pasted image 20261023164533.png|600]]
	4. ![[Pasted image 20261023164559.png|600]]
2. BFS
	1. ==BFS tells us the *shortest* path from node $a$ to node $b$.== Not good for wide graph.
	2. iterates over all of its adjacency linked list, i.e., neighbors at same level, thus breadth
	3. implementation
		1. Queue for tracking exploration
		2. Marking of *explored*
	4. Complexity for undirected
		1.  $V + 2E$: once every vertex, twice every edge
		2. it's twice because once from each end node. The 2nd time the other node won't be visited because it's status is already 'visited'.
3. DFS
	1. ==DFS tells us if a path from $a$ to $b$ even exists,== Not good for finding shortest path.
		1. E.g., Topological Sort (Sect 8.5): DAG may have many topological orderings, and DFS finds *one* of such orderings in $O(m+n)$ time 
	2. DFS will recurse on the 1st element and put rest of the linked list "on hold" until backtracking, therefore prioritize depth
	3. implementation
		1. Stack (*iterative*) or recursion. Recursion also leverages stack, just inexplicitly.
		2. Marking of *explored*
	4. Complexity for undirected
### References
1. [[Algorithms Illuminated (Part 2).pdf]] Chapter 7, 8
2. [Post](https://medium.com/basecs/from-theory-to-practice-representing-graphs-cfd782c5be38) by Vaidehi Joshi
### Coding
#### [542. 01 Matrix](https://leetcode.com/problems/01-matrix/) $\star$
1. I spent a lot of time try to solve it with DFS and failed--when use `visited` to avoid repeated visiting, it also block out potentially shorter paths. 
	1. E.g., `[[1, 1, 1], [0, 1, 1]]`. If I first went `(1, 2) - (0, 2) - (0, 1) - (1, 1) - (1, 0)` and have `(1, 1)` added to `visited`, then I'd never have the best path `(1, 2) - (1, 1) - (1, 0)`./
	2. ==🟡For shortest paths, use BFS!==
2. find *all possible starting points*, then run BFS. See [solution](https://youtu.be/YTnYte6U61w)
	1. This can be thought of as all starting points are 1 step from a single super starting point.
	2. No `visited` needed; coz `-1` means "not discovered yet" already
		```python
		from collections import deque
		def updateMatrix(mat: list[list[int]]) -> list[list[int]]:
		    rows, cols = len(mat), len(mat[0])
		    distances = [[-1] * cols for _ in range(rows)]
		    queue = deque()
		
		    for r in range(rows):
		        for c in range(cols):
		            if mat[r][c] == 0:
		                distances[r][c] = 0
		                queue.append((r, c))
		
		    while queue:
		        r, c = queue.popleft()
		        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
		            nr, nc = r + dr, c + dc
		            if (0 <= nr < rows and 0 <= nc < cols
		                    and distances[nr][nc] == -1):
		                distances[nr][nc] = distances[r][c] + 1
		                queue.append((nr, nc))
		    return distances
		```
#### [133. Clone Graph](https://leetcode.com/problems/clone-graph/) (understanding)
1. DFS and update values dynamically. [solution](https://youtu.be/mQeF6bN8hMk)
#### [417. Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/description/) (understanding)
1. [Solution](https://youtu.be/_ufr9Yz25lw) provided both DFS and BFS, but I prefer BFS which makes more sense to me. 
	1. the DFS needs to start at specific rows/cols, the generic DFS does not work
	2. *The vanilla DFS* in this situation won't work
		1. Coz you may end up visiting a location over and over, infinite loop, before it's registered. See example in BFS below.
2. *The BFS*
	1. starts with points on the boundaries, think of the ocean as super starting points
	2. always have `visted` so that we do not duplicate efforts 
		1. and only when `pop` the element do we add it to `visited`!
	3. at each level, only add legit `(next_r, next_c)` to `q`
		1. Every node in `visited` will be valid. *This is like you can only explore connected next nodes.*
		2. invalid cells may get added at later levels. 
			1. E.g., `9` in `[[1,2,3,4],[12,13,14,5],[11,16,15,6],[10,9,8,7]]` will be ignored at step 1 but added at last step. This means `9` is not directly connected to `10` but can be reached via `5->6->7->8->9`.
#### [261. Graph Valid Tree](https://neetcode.io/problems/valid-tree) (premium)
1. For a graph to be a tree (undirected)
	1. every node is connected, i.e., a tree has only one component and one can traverse the whole tree starting any node
	2. no cycle: any two nodes have exactly one path
2. For undirected tree, it's a good idea to keep both directions in the adjacency list
3. Since we need to use `visited` to determine base case, the use of `prev` in `dfs(node, prev)` keeps track of where we came from
	1. this enables us to start from any node
	2. to start, use `dfs(node, -1)` since no node would have `-1`.
4. `n == len(visited)` makes sure every node is connected
5. Full code
	```python
	def is_valid_tree(n, edges):	
		# get adjacency list	
		adj = {i:[] for i in range(n)}
		for i, j in edges:
			adj[i].append(j)
			adj[j].append(i)
	
		visited = set()
		def dfs(node, prev):
			if node in visited:  # cycle exists
				return False
	
			visited.add(node)
			for nei in adj[node]:
				if nei == prev:  # since we cannot use `visited`
					continue
				if not dfs(nei, node):
					return False
			return True
	
		import random
		root = random.choice(range(n))  # can start at any node
		no_cycle = dfs(root, -1)  # -1 is outside of possible node value
		all_connected = n == len(visited)  # if tree, should be able to traverse all nodes
		return no_cycle and all_connected

	# example: not valid tree, cycle of 1->2->3->1
	n = 5
	edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]	  
	
	```

Tow-D Maze

## - Heap
### Learning
1. Useful for fast, repeated access to minimum or maximum elements in a dynamically changing dataset. E.g., top-K problems.
2. Heaps are just binary trees with certain types of invariants
	1. order invariant: Min-heap, Max-heap
	2. shape invariant: nearly complete. I.e., every level is complete except possibly the last one, which is filled strictly from lef to right with no gaps.
		1. this enables quick operations with child and parent indexing.
3. Time complexity
	1. Find max/min: `O(1)`
	2. Insert/remove: `O(k)`, k is the size of the heap
	3. heapify: `O(k)`, k is the size of the heap
#### [973. K Closest Points to Origin](https://leetcode.com/problems/k-closest-points-to-origin/description/) $\star$
1. Append `-dist` to leverage `minHeap`. See [solution](https://youtu.be/IGRUukbD6p8)
2. Time complexity: `n*log(k)` where `k` is the size of the heap.
3. Redo this to familiarize with syntax.
```python
def kClosest(self, points: list[list[int]], k: int) -> list[list[int]]:
	def dist(x, y):
		return x**2 + y**2
	
	import heapq
	heap = []
	for x, y in points:
		d = dist(x, y)
		if len(heap) < k:
			# negate `d` because it's min_heap by default
			# pop() removes the root, i.e., minimum of heap
			heapq.heappush(heap, (-d, x, y))
		else:
			heapq.heappushpop(heap, (-d, x, y))
	return [[x, y] for _, x, y in heap]
```
#### [23. Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/)
1. Each step use a heap to find the min pointer and insert one new pointer, both `log(k)` time.
2. [solution](https://youtu.be/RyrVWP76lVo)
	1. the tuple of `(node.val, i, node)` where `i` is the ith list; in case of same `node.val` of nodes
	2. use `res` and `cur` where `cur` is to append node to `res`; I keep forgetting to do so
### Data structure
### [138. Copy List with Random Pointer](https://leetcode.com/problems/copy-list-with-random-pointer/)
1. Cute problem
### [380. Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/)
1. I did not understand the problem, see [solution](https://youtu.be/j4KwhBziOpg) for explanation
	1. basically need to avoid any $O(n)$ ops such as `list(self.val_map)` which is required for `random.choice` for the `getRandom()`

### [146 LRU Cache](https://leetcode.com/problems/lru-cache/description/)
1. The pattern is `hash map` and `double linked list`. 
	1. it's helpful to have `self.head` and `self.tail` as dummy nodes
	2. they tend to initialize an empty list with `self.head.next, self.tail.prev = self.tail, self.head`
### All solutions of a maze (try both BFS and DFS)
1. This is DFS, the backtracking is explicit
	1. BFS is good to find shortest solution, but not all of them
2. The `return` without any value means go back to previous calling point; 
```python
def find_all_paths(maze, start, end, path, solutions, visited):
    x, y = start
    if not (0 <= x < len(maze) and 0 <= y < len(maze[0])):
        return
    if maze[x][y] == 1 or visited[x][y]:
        return
        
    # Move forward: add to path and mark as visited
    path.append((x, y))
    visited[x][y] = True

    if (x, y) == end:
        solutions.append(list(path))  # Make a copy of the current path
    else:
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            find_all_paths(maze, (x+dx, y+dy), end, path, solutions, visited)
    
    # Backtrack: remove from path and unmark as visited
    path.pop()
    visited[x][y] = False

# Example usage:
maze = [
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 1, 0],
    [0, 0, 0, 0]
]
start = (0, 0)
end = (3, 3)
solutions = find_all_paths(maze, start, end)
print(f"Found {len(solutions)} solutions.")
for path in solutions:
    print(path)
```
### [78. Subsets](https://leetcode.com/problems/subsets/) (important)
1. In this [solution](https://youtu.be/z4uA-09uX4w?list=PLV5qT67glKSErHD66rKTfqerMYz9OaTOs&t=780)
	1. I like the `subset.pop()` for explicit backtracking
	2. Either subset is duplicated at every level, or `subset.copy()` in base case. This is because `subset` is a pointer, all `subset` in `res` will point to the same thing if no `.copy()`.
	3. 
### [17. Letter Combinations of a Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)
1. I just wanted to show the solution (my online) where
	1. the base case is at a maximum instead of a minimum
	2. no `return` until base case


## References
1. https://www.techinterviewhandbook.org/
2. 小小福LeetCode
	1. Very good explanations. For e.g., his survey on recursion
	2. [youtube](https://www.youtube.com/@leetcode8851)
	3. [github](https://fufuleetcode.github.io/)
2. https://www.jiakaobo.com/leetcode.html
3. 