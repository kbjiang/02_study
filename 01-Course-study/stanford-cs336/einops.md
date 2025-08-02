# [tutorial](https://github.com/arogozhnikov/einops/tree/main) from the creator
## basics
### `einops.rearrange`
1. `rearrange` does NOT change number of elements, only the shape of the tensor
2. Composition and decomposition
	1. Composition leads to *less dimensions* on the rhs, vice versa for decomposition
		```python
		# or compose a new dimension of batch and width
		rearrange(ims, "b h w c -> h (b w) c")
		# how it's done with `numpy`
		# assert_array_equal(rearrange(ims, "b h w c -> h (b w) c"), np.transpose(ims, (1, 0, 2, 3)).reshape(s1, s0*s2, s3))
		```
	2. Try manual op with simple matrices to see element movement. 
		1. It's just *the indices of element being updated*
		2. `[[[e, e],[e, e]], [[i, i], [i, i]], [[o, o],[o, o]], [[p, p], [p, p]]` from `4x2x2` to `(2x2)x(2x2)`, i.e. `(b1 b2) h w -> (b2 h) (b1 w)` with `b1=2`.
3. Order of axes matters
	1. See e.g. `rearrange(ims, "b h w c -> h (b w) c")` vs `rearrange(ims, "b h w c -> h (w b) c")`. Latter did an extra transpose.
### `einops.reduce` and `einops.repeat`
1. Opposite of each other; both change number of elements
2. Repeat along a *new* axis vs an *existing* axis
	1. new: `repeat(ims[0], "h w c -> new_axis h w c", new_axis=5)`
	2. existing: `repeat(ims[0], "h w c -> (repeat h) w c", repeat=5)`
3. Order matters
	1. E.g., `repeat(ims[1], "h w c -> h (repeat w) c", repeat=3)` vs `repeat(ims[0], "h w c -> h (w repeat) c", repeat=3)`, latter repeats each pixel three times.
4. 
### `einsum`
1.  The output can be transposed, but the input should have the dimensions aligned. 
	1. This is Ok: `"d_model d_ff, ... d_ff -> ... d_model"` 
	2. This is Not: `"d_model d_ff, d_ff ... -> ... d_model"` 