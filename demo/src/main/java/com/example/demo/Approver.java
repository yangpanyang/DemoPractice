package com.example.demo;

// 审批者
public interface Approver {
    void setNext(Approver next);
    boolean approve(String request) throws Exception;
}
