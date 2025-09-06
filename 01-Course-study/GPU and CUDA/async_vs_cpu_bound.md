
# Async vs CPU-bound Tasks

## Key Concepts

### 1. What a Thread Is
- A **thread** is a sequence of instructions executed by a CPU.  
- The thread itself is just a series of instructions; it doesn’t inherently create parallel execution.

---

### 2. Async/Await
- **Async** allows a task to **yield** when waiting (I/O, network, file operations).  
- The CPU can then run other tasks while the current task is waiting.  
- Async is **not for parallelizing CPU-heavy computations**.

**Example in Python:**
```python
async def read_file_async(path):
    data = await aiofiles.open(path).read()
    return data
```
- Here, `await` lets other tasks run while waiting for I/O.  

---

### 3. CPU-heavy Tasks and Async
- If all instructions in a thread are CPU-heavy:
  - `async` doesn’t speed things up.
  - The event loop is blocked until the task finishes.
- For CPU-heavy work, **threads or processes** are needed for real parallelism.

**Analogy:**  
- Async = letting someone wait in line while they handle other things.  
- CPU-heavy = the person works nonstop and can’t do anything else.  

**Illustration: CPU-heavy tasks on a single-core CPU**
```
Time →
Task A (CPU-heavy)  ██████████████
Task B (CPU-heavy)        ██████████████
Task C (CPU-heavy)              ██████████████

Event loop blocked while any task runs
```

**Illustration: I/O-heavy tasks (async shines)**
```
Time →
Task A (waiting)  ███───███───
Task B (waiting)  ─███───███─
Task C (waiting)  ──███───███

Event loop switches between tasks while waiting
```

---

### 4. Single-core CPU Behavior
- **Single-core CPU** executes only **one instruction at a time**.  
- **Time slicing** gives the illusion of multitasking.  
- Async only helps if tasks yield (e.g., waiting); CPU-heavy tasks block the core.  

**Analogy:**  
- Single chef in a kitchen (single-core CPU):
  - Async: chef puts a pot on stove and does other tasks while waiting.  
  - CPU-heavy: chef chops nonstop; no multitasking possible.

---

### ✅ Takeaways
- **Async is for I/O-bound tasks**, not CPU-bound tasks.  
- **CPU-heavy tasks require true parallelism** (multi-threading or multi-processing).  
- On a single-core CPU, only one instruction executes at a time. Async does **not make CPU-bound tasks faster**.
