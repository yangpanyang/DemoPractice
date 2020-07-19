package com.example.demo;

import org.springframework.stereotype.Component;

@Component("wineSeller")
public class WineSeller implements Seller {
    @Override
    public void sell() {
        System.out.println("Wine Seller");
    }
}
