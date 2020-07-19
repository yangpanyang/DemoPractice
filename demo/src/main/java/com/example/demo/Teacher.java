package com.example.demo;

public class Teacher {
    String name, id, subject;

    Teacher() {}

    @Override
    public String toString() {
        return name + "-" + id + "-" + subject;
    }
}
