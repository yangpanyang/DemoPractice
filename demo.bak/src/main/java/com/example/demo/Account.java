package com.example.demo;

public class Account {
    private int balance;

    Account() {
        balance = 0;
    }

    public void deposit(int amount) {
        balance += amount;
    }

    public void withdraw(int amount) {
        balance -= amount;
    }

    int getBalance() {
        return balance;
    }
}
