from enum import Enum


class LayerTypes(Enum):
    """
    Bu sınıf katman tipini tutar katmanın tipine göre her
    katmanda farklı işlem yapmak isteyebilirsiniz diye eklenmiştir.
    """

    input_layer = "input"
    """
    Giriş katmanını temsil eder.
    """
    hidden_layer = "hidden"
    """
    Gizli katmanı temsil eder.
    """

    output_layer = "output"
    """
    Çıkış katmanını temsil eder.
    """
