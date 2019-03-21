## Question 7: Fun Pattern Question

You are given a sequence of bit of size 5 where 4 bits are 0 and one bit is 1. So here are all the valid sequences 10000, 01000, 00100, 00010, 00001. At every second, the 1 bit will have a shift either to the left or to the right. If the current sequence is 10000, the 1 bit can only shift to the right, so the next sequence must be 01000. Also for the sequence 00001, the next sequence can only be 00010. Assume at every second, you are only allowed to sample at one index, would you be able to come up with an algorithm that finds the index of the value 1 in a finite amount of time?

For example, see the following attempt to locate the index of the 1 bit:
```
time 0, actual sequence is 01000, sample index 0, returned 0;
time 1, actual sequence is 10000, sample index 1, returned 0;
...
```

### Thought Process
1. There's a 1/5 chance of getting 1 if the function randomly guesses the index every time. However, the function can take an infinite amount of time.
2. Some positions are less likely to contain 1 in the sequence. For example, the following shows the different ways the value 1 can move. However, if I try to guess where 1 will most likely be, it'll still take an infinite amount of time to find the index in the worst case.
```
| 1 | 0 | 0 | 0 | 0 |
   <- 1 way to get to index 0

| 0 | 1 | 0 | 0 | 0 |
    -> <- 2 ways to get to index 1. Same for index 2 and 3.

| 0 | 0 | 0 | 0 | 1 |
                -> 1 way to get to index 4.
```
3. The very left bit can only shift to the right, and the very right bit can shift only to the left. Using this information, I can start at one position and wait `index + 1
` time and advance to the next position if I don't see 1 in my current position.

#### Time complexity analysis
In the worst case, the function will wait `current index + 1` time and check the next index.
This will take ( n (n+1) / 2 ) - 1 checks which is O(n^2) time. For example, the worst case situation would be when the 1 goes back and forth on index 0 and 1. It would take this function 14 checks to return index 1.
```
| 1 | 0 | 0 | 0 | 0 |

| 0 | 1 | 0 | 0 | 0 |
```

#### Space complexity analysis
The only additional space I'm using are several variables to keep track of the current index and wait time.


### How to run the code.
```bash
python fun_pattern.py
```
