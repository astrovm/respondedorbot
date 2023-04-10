import pytest
from main import convert_to_command


def test_convert_to_command():
    # Test basic string
    msg_text1 = "h3llo W0RLD"
    expected1 = "/H3LLO_W0RLD"
    assert convert_to_command(msg_text1) == expected1

    # Test string with special characters
    msg_text2 = "hello! world? or... mmm ...bye."
    expected2 = "/HELLO_SIGNODEEXCLAMACION_WORLD_SIGNODEPREGUNTA_OR_PUNTOSSUSPENSIVOS_MMM_PUNTOSSUSPENSIVOS_BYE_PUNTO"
    assert convert_to_command(msg_text2) == expected2

    # Test string with consecutive spaces
    msg_text3 = "  hello   world "
    expected3 = "/HELLO_WORLD"
    assert convert_to_command(msg_text3) == expected3

    # Test string with emoji
    msg_text4 = "😄hello 😄 world"
    expected4 = "/CARA_SONRIENDO_CON_OJOS_SONRIENTES_HELLO_CARA_SONRIENDO_CON_OJOS_SONRIENTES_WORLD"
    assert convert_to_command(msg_text4) == expected4

    # Test string with accented characters and Ñ
    msg_text5 = "hola ñandú ñ"
    expected5 = "/HOLA_NIANDU_ENIE"
    assert convert_to_command(msg_text5) == expected5

    # Test string with new line
    msg_text6 = "hola\nlinea\n"
    expected6 = "/HOLA_LINEA"
    assert convert_to_command(msg_text6) == expected6

    # Test empty string
    msg_text7 = ""
    expected7 = "Invalid input. Usage: /comando <text>"
    assert convert_to_command(msg_text7) == expected7
