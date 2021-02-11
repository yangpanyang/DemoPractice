package com.example.demo;

public class Rectangle implements Shape {
    @Override
    public void draw() {
        System.out.println("Draw Rectangle");
    }

    @Override
    public String getName() {
        return "Rectangle";
    }
}
