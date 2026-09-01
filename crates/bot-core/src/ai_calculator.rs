//! Safe arithmetic evaluator shared by the AI `calculate` tool.

use num_bigint::BigInt;

use crate::locale::Locale;

const MAX_SYNTAX_NODES: usize = 200;
const MAX_INTEGER_EXPONENT: u32 = 100_000;

#[derive(Debug, Clone, PartialEq)]
enum Number {
    Integer(BigInt),
    Float(f64),
}

impl Number {
    fn as_float(&self) -> Result<f64, CalculationError> {
        match self {
            Self::Integer(value) => value
                .to_string()
                .parse::<f64>()
                .map_err(|_| CalculationError::Invalid),
            Self::Float(value) => Ok(*value),
        }
    }

    fn is_zero(&self) -> bool {
        match self {
            Self::Integer(value) => value == &BigInt::from(0),
            Self::Float(value) => *value == 0.0,
        }
    }

    fn render(self) -> String {
        match self {
            Self::Integer(value) => value.to_string(),
            Self::Float(value) if value.is_finite() => {
                let rounded = (value * 100_000_000.0).round() / 100_000_000.0;
                if rounded == 0.0 {
                    "0".to_owned()
                } else if rounded.fract() == 0.0 {
                    format!("{rounded:.0}")
                } else {
                    rounded.to_string()
                }
            }
            Self::Float(value) => value.to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operator {
    Add,
    Subtract,
    Multiply,
    Divide,
    FloorDivide,
    Modulo,
    Power,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Token {
    Number(String),
    Operator(Operator),
    LeftParen,
    RightParen,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CalculationError {
    Invalid,
    TooLong,
    Forbidden,
    ZeroDivision,
}

#[must_use]
pub fn calculate_expression(expression: &str, locale: Locale) -> String {
    if expression.is_empty() {
        return localized(
            locale,
            "no se proporciono una expresion",
            "no expression was provided",
        );
    }
    match tokenize(expression).and_then(|tokens| Parser::new(&tokens).parse()) {
        Ok(value) => value.render(),
        Err(CalculationError::Invalid) => match locale {
            Locale::Es => format!("expresión inválida: {expression}"),
            Locale::En => format!("invalid expression: {expression}"),
        },
        Err(CalculationError::TooLong) => localized(
            locale,
            "expresión demasiado larga",
            "expression is too long",
        ),
        Err(CalculationError::Forbidden) => match locale {
            Locale::Es => format!("expresión no permitida: {expression}"),
            Locale::En => format!("expression is not allowed: {expression}"),
        },
        Err(CalculationError::ZeroDivision) => localized(
            locale,
            "no se puede dividir por cero",
            "cannot divide by zero",
        ),
    }
}

fn localized(locale: Locale, spanish: &str, english: &str) -> String {
    match locale {
        Locale::Es => spanish.to_owned(),
        Locale::En => english.to_owned(),
    }
}

fn tokenize(expression: &str) -> Result<Vec<Token>, CalculationError> {
    let chars = expression.chars().collect::<Vec<_>>();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        match chars[index] {
            value if value.is_whitespace() => index += 1,
            value if value.is_ascii_digit() || value == '.' => {
                let start = index;
                index += 1;
                while index < chars.len()
                    && (chars[index].is_ascii_digit()
                        || matches!(chars[index], '.' | 'e' | 'E')
                        || (matches!(chars[index], '+' | '-')
                            && matches!(chars.get(index.wrapping_sub(1)), Some('e' | 'E'))))
                {
                    index += 1;
                }
                tokens.push(Token::Number(chars[start..index].iter().collect()));
            }
            '+' => {
                tokens.push(Token::Operator(Operator::Add));
                index += 1;
            }
            '-' => {
                tokens.push(Token::Operator(Operator::Subtract));
                index += 1;
            }
            '*' if chars.get(index + 1) == Some(&'*') => {
                tokens.push(Token::Operator(Operator::Power));
                index += 2;
            }
            '*' => {
                tokens.push(Token::Operator(Operator::Multiply));
                index += 1;
            }
            '/' if chars.get(index + 1) == Some(&'/') => {
                tokens.push(Token::Operator(Operator::FloorDivide));
                index += 2;
            }
            '/' => {
                tokens.push(Token::Operator(Operator::Divide));
                index += 1;
            }
            '%' => {
                tokens.push(Token::Operator(Operator::Modulo));
                index += 1;
            }
            '(' => {
                tokens.push(Token::LeftParen);
                index += 1;
            }
            ')' => {
                tokens.push(Token::RightParen);
                index += 1;
            }
            _ => return Err(CalculationError::Forbidden),
        }
        if tokens.len() > MAX_SYNTAX_NODES {
            return Err(CalculationError::TooLong);
        }
    }
    Ok(tokens)
}

struct Parser<'a> {
    tokens: &'a [Token],
    index: usize,
    nodes: usize,
}

impl<'a> Parser<'a> {
    const fn new(tokens: &'a [Token]) -> Self {
        Self {
            tokens,
            index: 0,
            nodes: 0,
        }
    }

    fn parse(mut self) -> Result<Number, CalculationError> {
        let value = self.parse_additive()?;
        if self.index != self.tokens.len() {
            return Err(CalculationError::Invalid);
        }
        Ok(value)
    }

    fn parse_additive(&mut self) -> Result<Number, CalculationError> {
        let mut value = self.parse_multiplicative()?;
        while let Some(operator @ (Operator::Add | Operator::Subtract)) = self.operator() {
            self.index += 1;
            value = apply(operator, value, self.parse_multiplicative()?)?;
        }
        Ok(value)
    }

    fn parse_multiplicative(&mut self) -> Result<Number, CalculationError> {
        let mut value = self.parse_unary()?;
        while let Some(
            operator @ (Operator::Multiply
            | Operator::Divide
            | Operator::FloorDivide
            | Operator::Modulo),
        ) = self.operator()
        {
            self.index += 1;
            value = apply(operator, value, self.parse_unary()?)?;
        }
        Ok(value)
    }

    fn parse_unary(&mut self) -> Result<Number, CalculationError> {
        match self.operator() {
            Some(Operator::Add) => {
                self.index += 1;
                self.bump_node()?;
                self.parse_unary()
            }
            Some(Operator::Subtract) => {
                self.index += 1;
                self.bump_node()?;
                negate(self.parse_unary()?)
            }
            _ => self.parse_power(),
        }
    }

    fn parse_power(&mut self) -> Result<Number, CalculationError> {
        let value = self.parse_primary()?;
        if self.operator() == Some(Operator::Power) {
            self.index += 1;
            return apply(Operator::Power, value, self.parse_unary()?);
        }
        Ok(value)
    }

    fn parse_primary(&mut self) -> Result<Number, CalculationError> {
        self.bump_node()?;
        match self.tokens.get(self.index) {
            Some(Token::Number(raw)) => {
                self.index += 1;
                parse_number(raw)
            }
            Some(Token::LeftParen) => {
                self.index += 1;
                let value = self.parse_additive()?;
                if self.tokens.get(self.index) != Some(&Token::RightParen) {
                    return Err(CalculationError::Invalid);
                }
                self.index += 1;
                Ok(value)
            }
            _ => Err(CalculationError::Invalid),
        }
    }

    fn operator(&self) -> Option<Operator> {
        match self.tokens.get(self.index) {
            Some(Token::Operator(operator)) => Some(*operator),
            _ => None,
        }
    }

    fn bump_node(&mut self) -> Result<(), CalculationError> {
        self.nodes += 1;
        if self.nodes > MAX_SYNTAX_NODES {
            Err(CalculationError::TooLong)
        } else {
            Ok(())
        }
    }
}

fn parse_number(raw: &str) -> Result<Number, CalculationError> {
    if raw.chars().any(|value| matches!(value, '.' | 'e' | 'E')) {
        raw.parse::<f64>()
            .map(Number::Float)
            .map_err(|_| CalculationError::Invalid)
    } else {
        raw.parse::<BigInt>()
            .map(Number::Integer)
            .map_err(|_| CalculationError::Invalid)
    }
}

fn negate(value: Number) -> Result<Number, CalculationError> {
    Ok(match value {
        Number::Integer(value) => Number::Integer(-value),
        Number::Float(value) => Number::Float(-value),
    })
}

fn apply(operator: Operator, left: Number, right: Number) -> Result<Number, CalculationError> {
    match operator {
        Operator::Add => integer_or_float(left, right, |a, b| a + b, |a, b| a + b),
        Operator::Subtract => integer_or_float(left, right, |a, b| a - b, |a, b| a - b),
        Operator::Multiply => integer_or_float(left, right, |a, b| a * b, |a, b| a * b),
        Operator::Divide => divide(left, right),
        Operator::FloorDivide => floor_divide(left, right),
        Operator::Modulo => modulo(left, right),
        Operator::Power => power(left, right),
    }
}

fn integer_or_float(
    left: Number,
    right: Number,
    integer: impl FnOnce(BigInt, BigInt) -> BigInt,
    float: impl FnOnce(f64, f64) -> f64,
) -> Result<Number, CalculationError> {
    match (left, right) {
        (Number::Integer(left), Number::Integer(right)) => {
            Ok(Number::Integer(integer(left, right)))
        }
        (left, right) => Ok(Number::Float(float(left.as_float()?, right.as_float()?))),
    }
}

fn divide(left: Number, right: Number) -> Result<Number, CalculationError> {
    if right.is_zero() {
        return Err(CalculationError::ZeroDivision);
    }
    Ok(Number::Float(left.as_float()? / right.as_float()?))
}

fn floor_divide(left: Number, right: Number) -> Result<Number, CalculationError> {
    if right.is_zero() {
        return Err(CalculationError::ZeroDivision);
    }
    match (left, right) {
        (Number::Integer(left), Number::Integer(right)) => {
            let quotient = &left / &right;
            let remainder = &left % &right;
            let opposite_signs = (left < BigInt::from(0)) != (right < BigInt::from(0));
            Ok(Number::Integer(
                if remainder != BigInt::from(0) && opposite_signs {
                    quotient - 1
                } else {
                    quotient
                },
            ))
        }
        (left, right) => Ok(Number::Float(
            (left.as_float()? / right.as_float()?).floor(),
        )),
    }
}

fn modulo(left: Number, right: Number) -> Result<Number, CalculationError> {
    if right.is_zero() {
        return Err(CalculationError::ZeroDivision);
    }
    match (left, right) {
        (Number::Integer(left), Number::Integer(right)) => {
            let remainder = &left % &right;
            let opposite_signs = (remainder < BigInt::from(0)) != (right < BigInt::from(0));
            Ok(Number::Integer(
                if remainder != BigInt::from(0) && opposite_signs {
                    remainder + right
                } else {
                    remainder
                },
            ))
        }
        (left, right) => {
            let left = left.as_float()?;
            let right = right.as_float()?;
            Ok(Number::Float(left - (left / right).floor() * right))
        }
    }
}

fn power(left: Number, right: Number) -> Result<Number, CalculationError> {
    if let (Number::Integer(left), Number::Integer(right)) = (&left, &right)
        && right >= &BigInt::from(0)
    {
        let exponent = right
            .to_string()
            .parse::<u32>()
            .map_err(|_| CalculationError::TooLong)?;
        if exponent > MAX_INTEGER_EXPONENT {
            return Err(CalculationError::TooLong);
        }
        return Ok(Number::Integer(left.pow(exponent)));
    }
    Ok(Number::Float(left.as_float()?.powf(right.as_float()?)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates_precedence_parentheses_powers_and_exact_large_integers() {
        assert_eq!(calculate_expression("2 + 3 * 4", Locale::En), "14");
        assert_eq!(calculate_expression("(2 + 3) * 4", Locale::En), "20");
        assert_eq!(calculate_expression("2 ** 10", Locale::En), "1024");
        assert_eq!(calculate_expression("2 ** -2", Locale::En), "0.25");
        assert_eq!(
            calculate_expression("100000000000000000000 + 1", Locale::En),
            "100000000000000000001"
        );
    }

    #[test]
    fn matches_python_division_floor_modulo_and_float_rounding() {
        assert_eq!(calculate_expression("5 / 2", Locale::En), "2.5");
        assert_eq!(calculate_expression("-5 // 2", Locale::En), "-3");
        assert_eq!(calculate_expression("5 // -2", Locale::En), "-3");
        assert_eq!(calculate_expression("-5 % 2", Locale::En), "1");
        assert_eq!(calculate_expression("5 % -2", Locale::En), "-1");
        assert_eq!(calculate_expression("1 / 3", Locale::En), "0.33333333");
        assert_eq!(calculate_expression("4 // 2", Locale::En), "2");
        assert_eq!(calculate_expression("5 % 2", Locale::En), "1");
        assert_eq!(calculate_expression("4.0 // 1.5", Locale::En), "2");
        assert_eq!(calculate_expression("5.5 % 2", Locale::En), "1.5");
        assert_eq!(calculate_expression("1.5 + 2.25", Locale::En), "3.75");
        assert_eq!(calculate_expression("3.5 - 1.25", Locale::En), "2.25");
        assert_eq!(calculate_expression("1.5 * 2", Locale::En), "3");
        assert_eq!(calculate_expression("1e3 / 4", Locale::En), "250");
        assert_eq!(calculate_expression("+2", Locale::En), "2");
        assert_eq!(calculate_expression("-1.5", Locale::En), "-1.5");
        assert_eq!(calculate_expression("0.0 / 2", Locale::En), "0");
    }

    #[test]
    fn localizes_missing_invalid_forbidden_zero_and_bounded_inputs() {
        assert_eq!(
            calculate_expression("", Locale::En),
            "no expression was provided"
        );
        assert_eq!(
            calculate_expression("1 +", Locale::Es),
            "expresión inválida: 1 +"
        );
        assert_eq!(
            calculate_expression("open('/tmp/x')", Locale::En),
            "expression is not allowed: open('/tmp/x')"
        );
        assert_eq!(
            calculate_expression("1 / 0", Locale::Es),
            "no se puede dividir por cero"
        );
        assert_eq!(
            calculate_expression("1.0 / 0.0", Locale::En),
            "cannot divide by zero"
        );
        assert_eq!(
            calculate_expression("1 // 0", Locale::En),
            "cannot divide by zero"
        );
        assert_eq!(
            calculate_expression("1 % 0", Locale::En),
            "cannot divide by zero"
        );
        assert_eq!(
            calculate_expression("(1 + 2", Locale::En),
            "invalid expression: (1 + 2"
        );
        assert_eq!(
            calculate_expression("1 2", Locale::En),
            "invalid expression: 1 2"
        );
        assert_eq!(
            calculate_expression(".", Locale::En),
            "invalid expression: ."
        );
        assert_eq!(
            calculate_expression("open('/tmp/x')", Locale::Es),
            "expresión no permitida: open('/tmp/x')"
        );
        assert_eq!(
            calculate_expression(&"1+".repeat(201), Locale::En),
            "expression is too long"
        );
        assert_eq!(
            calculate_expression("2 ** 100001", Locale::En),
            "expression is too long"
        );
    }
}
