package com.example.demo;

public class BoA {
    String v1, v2, v3;

    BoA() { }

    BoA(String v1, String v2, String v3) {
        this.v1 = v1;
        this.v2 = v2;
        this.v3 = v3;
    }

    @Override
    public String toString() {
        return "A: " + v1 + ", " + v2 + ", " + v3;
    }
}
