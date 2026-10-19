"""Target schema: the scalar money fields of a CORD-v2 receipt."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Receipt(BaseModel):
    """Every field is optional: ``None`` means "not printed on this receipt"."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    total_price: str | None = Field(
        None, description="Final amount due after tax, service and discount (TOTAL, GRAND TOTAL)."
    )
    subtotal_price: str | None = Field(
        None, description="Sum of the items before tax, service and discount (SUBTOTAL)."
    )
    tax_price: str | None = Field(None, description="Tax amount (TAX, VAT, PB1, PPN).")
    service_price: str | None = Field(None, description="Service charge amount (SERVICE, SC).")
    discount_price: str | None = Field(
        None, description="Discount on the whole bill (DISCOUNT, DISC)."
    )
    cash_price: str | None = Field(None, description="Amount paid in cash (CASH, TUNAI).")
    change_price: str | None = Field(
        None, description="Change given back to the customer (CHANGE, KEMBALI)."
    )
    credit_card_price: str | None = Field(
        None, description="Amount paid by card (CARD, CREDIT, DEBIT, VISA, MASTER)."
    )

    @field_validator("*", mode="before")
    @classmethod
    def _to_text(cls, value: object) -> str | None:
        # Models return numbers, strings, or the odd list; scoring compares text.
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, list) and len(value) == 1:
            return cls._to_text(value[0])
        return str(value)


FIELDS: tuple[str, ...] = tuple(Receipt.model_fields)

# CORD-v2 ``gt_parse`` location of each schema field.
CORD_KEYS: dict[str, tuple[str, str]] = {
    "total_price": ("total", "total_price"),
    "subtotal_price": ("sub_total", "subtotal_price"),
    "tax_price": ("sub_total", "tax_price"),
    "service_price": ("sub_total", "service_price"),
    "discount_price": ("sub_total", "discount_price"),
    "cash_price": ("total", "cashprice"),
    "change_price": ("total", "changeprice"),
    "credit_card_price": ("total", "creditcardprice"),
}
