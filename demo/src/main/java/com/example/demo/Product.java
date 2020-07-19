package com.example.demo;

public class Product {
    String brand;
    Integer id;
    Float price;

    Product(String brand, Integer id, Float price) {
        this.brand = brand;
        this.id = id;
        this.price = price;
    }

    @Override
    public String toString() {
        return brand + "/" + id + ": " + price;
    }

    public void showInfo() {
        System.out.println(toString());
    }

    public String getBrand() {
        return brand;
    }

    public void setBrand(String brand) {
        this.brand = brand;
    }

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public Float getPrice() {
        return price;
    }

    public void setPrice(Float price) {
        this.price = price;
    }
}
