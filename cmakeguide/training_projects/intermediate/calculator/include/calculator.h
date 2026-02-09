#ifndef CALCULATOR_H
#define CALCULATOR_H

#include <string>
#include <vector>

namespace calc {
    class Calculator {
    public:
        Calculator();
        ~Calculator();
        
        double add(double a, double b);
        double subtract(double a, double b);
        double multiply(double a, double b);
        double divide(double a, double b);
        double power(double base, double exponent);
        double sqrt(double value);
        
        // Expression evaluation
        double evaluate_expression(const std::string& expr);
        
        // History functions
        std::vector<std::string> get_history() const;
        void clear_history();
        
    private:
        std::vector<std::string> history_;
        void record_operation(const std::string& operation, double result);
    };
}

#endif