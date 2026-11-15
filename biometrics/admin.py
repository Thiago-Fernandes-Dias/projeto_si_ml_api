from django.contrib import admin

from biometrics.models import Amostra, ModeloBiometrico, User


@admin.register(User)
class UsuarioAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "nome",
        "biometria_cadastrada_em",
        "tentativas_falhas",
        "bloqueado_ate",
        "criado_em",
    )
    search_fields = ("nome",)
    readonly_fields = ("id", "criado_em")


@admin.register(Amostra)
class AmostraAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "usuario",
        "origem",
        "repeticao",
        "decisao",
        "confianca",
        "modelo",
        "coletada_em",
    )
    list_filter = ("origem", "decisao", "coletada_em")
    search_fields = ("usuario__nome",)
    readonly_fields = (
        "id",
        "usuario",
        "origem",
        "vetor",
        "repeticao",
        "duracao_total_ms",
        "decisao",
        "confianca",
        "modelo",
        "coletada_em",
    )

    def has_add_permission(self, request: object) -> bool:
        return False

    def has_change_permission(self, request: object, obj: object = None) -> bool:
        return False

    def has_delete_permission(self, request: object, obj: object = None) -> bool:
        return False


@admin.register(ModeloBiometrico)
class ModeloBiometricoAdmin(admin.ModelAdmin):
    list_display = ("id", "usuario", "versao", "algoritmo", "caminho_arquivo", "criado_em")
    list_filter = ("algoritmo", "criado_em")
    search_fields = ("usuario__nome", "caminho_arquivo")
    readonly_fields = ("criado_em",)
