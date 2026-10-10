# Virtual Memory Paging

Operating systems give each process a contiguous virtual address space backed by physical frames allocated on demand. Page tables map virtual pages to frames; page faults fetch missing pages from disk. Paging eliminates external fragmentation because no physical region must be contiguous, at the cost of translation overhead hidden by TLB caches. Demand paging, copy-on-write, and mapped files all build on the same page-granularity abstraction.
