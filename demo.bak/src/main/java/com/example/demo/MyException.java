package com.example.demo;

public class MyException extends Exception {
    private String msg;

    MyException(String msg) {
        this.msg = msg;
    }

    @Override
    public String toString() {
        return super.toString() + ": " + msg;
    }
}
