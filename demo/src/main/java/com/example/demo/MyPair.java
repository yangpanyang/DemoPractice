package com.example.demo;

public class MyPair <T1, T2> {
    T1 key;
    T2 value;

    MyPair(T1 key, T2 value) {
        this.key = key;
        this.value = value;
    }

    public void setKey(T1 key) {
        this.key = key;
    }

    public void setValue(T2 value) {
        this.value = value;
    }

    @Override
    public String toString() {
        return key.toString() + ": " + value.toString();
    }
}
