package com.example.demo;

public class Subscriber {
    String name;

    Subscriber(String name) { this.name = name; }

    public String getName() { return name; }

    public void register(MessageCenter messageCenter) {
        messageCenter.register(this);
    }

    public void unregister(MessageCenter messageCenter) {
        messageCenter.unregister(this);
    }

    public void notify(String message) {
        System.out.println(name + " get notify: " + message);
    }

}
