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

## REST
## SOAP
## RPC
## GraphQL

# Questions
## Difference between Cookies and cache?
Cookies generally on the server side. while cache is on the client side.
When we go through any website cookies remembers all the pages we are visiting. So that if someone will come next time on their website they don't have to load all those things again. (It will improve the engagement, user experience)
When we visit any website they will load the html css for that page in browser. So that when we will visit that website again we don't have to wait to load those things again. as those things are already present in our browser. 

## Difference between REST, SOAP, RPC & GraphQL?
