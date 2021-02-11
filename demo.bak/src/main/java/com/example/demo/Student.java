package com.example.demo;

public class Student {
    String name, id;
    Integer age;

    Student() {}

    @Override
    public String toString() {
        return name + "-" + id + "-" + age;
    }
}
