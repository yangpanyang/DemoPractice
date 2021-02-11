package com.example.demo;

public class Circle implements Shape {
    @Override
    public void draw() {
        System.out.println("Draw Circle");
    }

    @Override
    public String getName() {
        return "Circle";
    }
}
