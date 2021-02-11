package com.example.demo;

import java.util.ArrayList;
import java.util.List;

public class MessageCenter {
    List<Subscriber> subscriberList;

    MessageCenter() {
        subscriberList = new ArrayList<>();
    }

    public void register(Subscriber subscriber) {
        subscriberList.add(subscriber);
    }

    public void unregister(Subscriber subscriber) {
        subscriberList.remove(subscriber);
    }

    public void notify(String message) {
        for (Subscriber subscriber : subscriberList) {
            subscriber.notify(message);
        }
    }
}
