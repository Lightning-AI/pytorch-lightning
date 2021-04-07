{{ name | escape | underline }}

.. currentmodule:: {{ fullname }}

{% block functions %}
{% if functions %}
.. rubric:: Functions

.. autosummary::
    :nosignatures:
{% for item in functions %}
    {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}
.. rubric:: Classes

.. autosummary::
    :nosignatures:
{% for item in classes %}
    {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block exceptions %}
{% if exceptions %}
.. rubric:: Exceptions

.. autosummary::
    :nosignatures:
{% for item in exceptions %}
    {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

.. automodule:: {{ fullname }}
