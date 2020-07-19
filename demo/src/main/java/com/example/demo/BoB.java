package com.example.demo;

public class BoB {
    String v1, v2;

    BoB() { }

    BoB(String v1, String v2) {
        this.v1 = v1;
        this.v2 = v2;
    }

    @Override
    public String toString() {
        return "B: " + v1 + ", " + v2;
    }
}
