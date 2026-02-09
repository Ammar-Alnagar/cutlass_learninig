#include "calculator.h"
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace calc {
    Calculator::Calculator() = default;
    
    Calculator::~Calculator() = default;
    
    double Calculator::add(double a, double b) {
        double result = a + b;
        record_operation(std::to_string(a) + " + " + std::to_string(b), result);
        return result;
    }
    
    double Calculator::subtract(double a, double b) {
        double result = a - b;
        record_operation(std::to_string(a) + " - " + std::to_string(b), result);
        return result;
    }
    
    double Calculator::multiply(double a, double b) {
        double result = a * b;
        record_operation(std::to_string(a) + " * " + std::to_string(b), result);
        return result;
    }
    
    double Calculator::divide(double a, double b) {
        if (b == 0) {
            throw std::domain_error("Division by zero");
        }
        double result = a / b;
        record_operation(std::to_string(a) + " / " + std::to_string(b), result);
        return result;
    }
    
    double Calculator::power(double base, double exponent) {
        double result = std::pow(base, exponent);
        record_operation(std::to_string(base) + " ^ " + std::to_string(exponent), result);
        return result;
    }
    
    double Calculator::sqrt(double value) {
        if (value < 0) {
            throw std::domain_error("Square root of negative number");
        }
        double result = std::sqrt(value);
        record_operation("sqrt(" + std::to_string(value) + ")", result);
        return result;
    }
    
    double Calculator::evaluate_expression(const std::string& expr) {
        // Simplified expression evaluator for basic operations
        // In a real implementation, this would be more sophisticated
        return 0.0; // Placeholder
    }
    
    std::vector<std::string> Calculator::get_history() const {
        return history_;
    }
    
    void Calculator::clear_history() {
        history_.clear();
    }
    
    void Calculator::record_operation(const std::string& operation, double result) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << result;
        history_.push_back(operation + " = " + oss.str());
    }
}