package com.example.demo;

public class ShapeFillShadow implements Shape {
    Shape shape;
    String shadow;

    ShapeFillShadow(Shape shape, String shadow) {
        this.shape = shape;
        this.shadow = shadow;
    }

    void fillShadow() {
        System.out.println("Fill " + shadow + " for " + shape.getName());
    }

    @Override
    public void draw() {
        shape.draw();
        fillShadow();
    }

    @Override
    public String getName() {
        return "FillShadow + " + shape.getName();
    }
}
