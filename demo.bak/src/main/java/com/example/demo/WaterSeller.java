package com.example.demo;

import org.springframework.stereotype.Component;

@Component("waterSeller")
public class WaterSeller implements Seller {
    @Override
    public void sell() {
        System.out.println("Water Seller");
    }
}
