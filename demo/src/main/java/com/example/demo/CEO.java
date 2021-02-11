package com.example.demo;

public class CEO implements Approver {
    CEO() {}

    @Override
    public void setNext(Approver next) { }

    @Override
    public boolean approve(String request) throws Exception {
        if (request.startsWith("C:")) {
            return true;
        } else {
            return false;
        }
        // return request.startsWith("C:");
    }
}
