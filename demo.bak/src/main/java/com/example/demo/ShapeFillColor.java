package com.example.demo;

public class ShapeFillColor implements Shape{
    Shape shape;
    String color;

    ShapeFillColor(Shape shape, String color) {
        this.shape = shape;
        this.color = color;
    }

    void fillColor() {
        System.out.println("Fill " + color + " for " + shape.getName());
    }

    @Override
    public void draw() {
        shape.draw();
        fillColor();
    }

    @Override
    public String getName() {
        return "FillColor " + shape.getName();
    }
}
