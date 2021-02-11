package com.example.demo;

public class WineSellerDI implements Seller {
    @Override
    public void sell() {
        System.out.println("Wine Seller Dependency Injection");
    }
}
