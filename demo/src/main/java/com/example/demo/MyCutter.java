package com.example.demo;

import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.*;
import org.springframework.stereotype.Component;

@Component  // 进入main函数之前，都会自动的实例化，不用显示的生成这个类
@Aspect  // 代表要实现切的功能了
public class MyCutter {
    // 定义一个切入点：执行com.example.demo.Calc.div这个方法时，并且参数是(int, int)，就切这个点
    // 通过注解实现：通过注解知道，当你执行div方法的时候，会把相关的切面代码放进来
    // execution参数格式为：包名.类名.函数名(参数)，"*"通配符，"..."匹配任意参数
    // 还可以用target/within处理其它情况
    @Pointcut("execution(* com.example.demo.Calc.div(int, int))")
    public void div() {
    } // 这个函数的作用类似给切点起个名字

    // 函数div执行前，把代码放进来，参数是通过JoinPoint传递的
    @Before("div()")
    public void before(JoinPoint joinPoint) {
        Object[] args = joinPoint.getArgs();
        // printf格式化打印
        System.out.printf("Before %s: %d, %d\n", joinPoint.getSignature().getName(), args[0], args[1]);
    }

    // 直接执行这个函数
    @Around("div()")
    public Object around(ProceedingJoinPoint pjp) throws Throwable {
        System.out.printf("Entering %s\n", pjp.getSignature().getName());
        Object result = pjp.proceed();
        System.out.printf("Leaving %s\n", pjp.getSignature().getName());
        return result;
    }

    // 在返回之后再做一些事情，即在after之后执行，用pointcut指定切面，绑定返回值
    @AfterReturning(pointcut = "div()", returning = "ret")
    public void afterReturning(Object ret) {
        System.out.printf("AfterReturning, return value is %s\n", ret.toString());
    }

    // 在around内被调用，around之后执行
    @After(value = "div()")
    public void after(JoinPoint joinPoint) {
        System.out.printf("After %s\n", joinPoint.getSignature().getName());
    }

    // 处理异常，需要的话重新包装一下再扔出，默认扔出获得的异常
    @AfterThrowing(pointcut = "div()", throwing = "e")
    public void afterThrowing(JoinPoint joinPoint, Exception e) throws Throwable {
        // e.getMessage()实现把异常打印出来
        System.out.printf("AfterThrowing %s: %s\n", joinPoint.getSignature().getName(), e.getMessage());
        // throws new Exception("Fuck"); // 看下此时的异常信息
    }
}
