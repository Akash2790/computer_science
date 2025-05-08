# System-design
Here I will add daily one new topic with proper explanation.

## TCP
When we send the date over the network like files, they're broken down into packets and then send over the internet and when they arrived into the destination, the packets are numbered so that they can be reassembeled into the correct order. if some packets are broken or missing, TCP ensures that they'll be resent. This is what makes it a reliable protocol and that why many other protocols like HTTP and websockets are built on top of TCP.

## DNS

## Caching

## Load Balancer

## Authorisation 

## Authentication

## Reverse Proxy

## API Gateway

## JWT
These Tokens starts with "ey" ex: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"

## Request Header: 
We can think it like we ship label put on a package, it tells us where the package is going, who its from and maybe some other metadata about the package itself
Request Body: This is basically the package content. to test it out we can open our dev tools from browser and under network tab 

Even in HTTP there is some API pattern that we could follow. Like the most popular one is Rest API, GraphQL

## REST
## SOAP
## RPC
## GraphQL
## Sharding: 
When single database is not enough to handle the load. Then  we create multiple/distrubute database to handle the load and this process is called sharding. Spliting your big data into mutually subsets of data and distributing it. 

# Questions
## Difference between Cookies and cache?
Cookies generally on the server side. while cache is on the client side.
When we go through any website cookies remembers all the pages we are visiting. So that if someone will come next time on their website they don't have to load all those things again. (It will improve the engagement, user experience)
When we visit any website they will load the html css for that page in browser. So that when we will visit that website again we don't have to wait to load those things again. as those things are already present in our browser. 

## Difference between REST, SOAP, RPC & GraphQL?

## Difference between Router and Load Balancer?
### **Router**
**Primary Function:** 
Route traffic between different networks.

**Decision Making:** 
Choose the best path for packets to travel based on their destination IP address.

**Example**: Connecting your home network to the internet. 

### **Load Balancers**
**Primary Function:** Distribute traffic across multiple servers. 

**Decision Making:** Based on various factors, such as server health, to determine which server to send traffic to. 

**Example**: Spreading traffic across multiple web servers to prevent any one server from being overloaded. 

## Single Thread Vs Multi Thread
In Easy language, Single thread means one instruction will be executed at a time. Multi thread means multiple instructions will be executed at a time.

Single-threaded programming languages are often straightforward and easy to work with. They are well-suited for tasks that do not require parallelism or when the tasks can be efficiently handled sequentially. These languages are particularly useful for simple scripts, web development, and applications where performance gains from parallelism are not critical.

Multi-threaded programming languages offer the advantage of parallel execution, allowing tasks to be performed concurrently. This can lead to significant performance improvements, especially in tasks that involve heavy computation, data processing, or network operations. Multi-threading is well-suited for applications that can be divided into independent sub-tasks that can be executed simultaneously.

**Single-Threaded Language Example** :: Python, Javascript, Angular

**Multi-Threaded Language Example** :: Java, C#

### Is python Multithreaded?
Python's default interpreter (CPython) is single-threaded, meaning it can only execute one thread of Python code at a time due to the Global Interpreter Lock (GIL).
However, Python is multithreaded in the sense that it supports the creation and management of multiple threads. This means you can create multiple threads and have them appear to run concurrently, but only one thread can execute Python bytecode at a time. 

Python provides libraries (like the threading module) that allow you to create and manage multiple threads. These threads can appear to run concurrently, but only one can be actively executing Python code at any given moment due to the GIL. 
