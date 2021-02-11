package com.example.demo;

public class ApproverObject {
    Approver approver;

    boolean approve(String request) throws Exception {
        return approver.approve(request);
    }
}
