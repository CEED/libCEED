abstract type AbstractRequest end

struct RequestImmediate <: AbstractRequest end
Base.getindex(::RequestImmediate) = C.CEED_REQUEST_IMMEDIATE[]

struct RequestOrdered <: AbstractRequest end
Base.getindex(::RequestOrdered) = C.CEED_REQUEST_ORDERED[]

#=
# CeedRequest is not fully implemented in libCEED. When it is implemented, the
# following can be used as a starting point for the Julia interface.

struct Request <: AbstractRequest
    ref::RefValue{C.CeedRequest}
end

Request() = Request(Ref{C.CeedRequest}())

Base.wait(req::Request) = C.CeedRequestWait(req[])
=#
