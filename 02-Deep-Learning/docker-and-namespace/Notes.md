**Background**: Our fastapi app ran ~10x slower when containerized than on the bare-metal machine.  I set out to understand why and realized that I don't really understand how Docker works. And to understand how Docker works I need to understand *namespace* and *cgroups*.

### namespace, cgroup and Docker
1. Nice introduction on namespace. I am trying to follow all examples in it. 
	1. https://lwn.net/Articles/531114/
	2. supplementary video https://youtu.be/0kJPa-1FuoI
2. Explains how Docker works. https://youtu.be/-YnMr1lj4Z8
3. Explalins namespaces and cgroups. https://youtu.be/x1npPrzyKfs

### Docker performance analysis
1. From the author of FlameGraph. https://youtu.be/bK9A5ODIgac
2. A potential similar case. Resolved using `perf`. https://www.coonote.com/note/perf-iostat-performance-analysis.html 
3. If perf not working: https://blog.eastonman.com/blog/2021/02/use-perf/
4. Another possible similar case.

