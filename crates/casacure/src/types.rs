//! CASA value types and their numpy/python mapping.
//!
//! Mirrors the `_TABLE_TO_PY` / `_PY_TO_TABLE` mapping used by dask-ms
//! (`daskms/columns.py:15-54`, see `CASACORE_TO_CASA_RS.md` §4).

use thiserror::Error;

/// A CASA column value type.
///
/// The string forms accepted by [`ValueType::from_casa_name`] include every
/// alias python-casacore reports in column descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    /// `BOOL` / `BOOLEAN` — numpy `bool`
    Bool,
    /// `BYTE` / `UCHAR` — numpy `uint8`
    Byte,
    /// `SHORT` / `SMALLINT` — numpy `int16`
    Short,
    /// `USHORT` / `USMALLINT` — numpy `uint16`
    UShort,
    /// `INT` / `INTEGER` — numpy `int32`
    Int,
    /// `UINT` / `UINTEGER` — numpy `uint32`
    UInt,
    /// `FLOAT` — numpy `float32`
    Float,
    /// `DOUBLE` — numpy `float64`
    Double,
    /// `FCOMPLEX` / `COMPLEX` — numpy `complex64`
    Complex,
    /// `DCOMPLEX` — numpy `complex128`
    DComplex,
    /// `STRING` — numpy `object`
    String,
}

/// Errors from parsing type names.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("unknown CASA value type name: {0:?}")]
pub struct UnknownValueType(pub String);

impl ValueType {
    /// All value types, in declaration order.
    pub const ALL: [ValueType; 11] = [
        ValueType::Bool,
        ValueType::Byte,
        ValueType::Short,
        ValueType::UShort,
        ValueType::Int,
        ValueType::UInt,
        ValueType::Float,
        ValueType::Double,
        ValueType::Complex,
        ValueType::DComplex,
        ValueType::String,
    ];

    /// Canonical (uppercase) casacore name, as used in column descriptors.
    pub const fn casa_name(self) -> &'static str {
        match self {
            ValueType::Bool => "BOOL",
            ValueType::Byte => "BYTE",
            ValueType::Short => "SHORT",
            ValueType::UShort => "USHORT",
            ValueType::Int => "INT",
            ValueType::UInt => "UINT",
            ValueType::Float => "FLOAT",
            ValueType::Double => "DOUBLE",
            ValueType::Complex => "COMPLEX",
            ValueType::DComplex => "DCOMPLEX",
            ValueType::String => "STRING",
        }
    }

    /// Name as written into the binary `table.dat` header by casacore.
    ///
    /// These are the lowercase names casacore's `TypeName` registration uses.
    pub const fn table_dat_name(self) -> &'static str {
        match self {
            ValueType::Bool => "Bool",
            ValueType::Byte => "uChar",
            ValueType::Short => "Short",
            ValueType::UShort => "uShort",
            ValueType::Int => "Int",
            ValueType::UInt => "uInt",
            ValueType::Float => "float",
            ValueType::Double => "double",
            ValueType::Complex => "Complex",
            ValueType::DComplex => "DComplex",
            ValueType::String => "String",
        }
    }

    /// Parse any casacore type name or alias (case-insensitive).
    pub fn from_casa_name(name: &str) -> Result<Self, UnknownValueType> {
        let upper = name.to_ascii_uppercase();
        Ok(match upper.as_str() {
            "BOOL" | "BOOLEAN" => ValueType::Bool,
            "BYTE" | "UCHAR" => ValueType::Byte,
            "SHORT" | "SMALLINT" => ValueType::Short,
            "USHORT" | "USMALLINT" => ValueType::UShort,
            "INT" | "INTEGER" => ValueType::Int,
            "UINT" | "UINTEGER" => ValueType::UInt,
            "FLOAT" => ValueType::Float,
            "DOUBLE" => ValueType::Double,
            "FCOMPLEX" | "COMPLEX" => ValueType::Complex,
            "DCOMPLEX" => ValueType::DComplex,
            "STRING" => ValueType::String,
            _ => return Err(UnknownValueType(name.to_string())),
        })
    }

    /// numpy dtype name, matching dask-ms's `_TABLE_TO_PY` mapping.
    pub const fn numpy_name(self) -> &'static str {
        match self {
            ValueType::Bool => "bool",
            ValueType::Byte => "uint8",
            ValueType::Short => "int16",
            ValueType::UShort => "uint16",
            ValueType::Int => "int32",
            ValueType::UInt => "uint32",
            ValueType::Float => "float32",
            ValueType::Double => "float64",
            ValueType::Complex => "complex64",
            ValueType::DComplex => "complex128",
            ValueType::String => "object",
        }
    }

    /// Size of one element in bytes, or `None` for variable-size `String`.
    pub const fn element_size(self) -> Option<usize> {
        match self {
            ValueType::Bool => Some(1),
            ValueType::Byte => Some(1),
            ValueType::Short => Some(2),
            ValueType::UShort => Some(2),
            ValueType::Int => Some(4),
            ValueType::UInt => Some(4),
            ValueType::Float => Some(4),
            ValueType::Double => Some(8),
            ValueType::Complex => Some(8),
            ValueType::DComplex => Some(16),
            ValueType::String => None,
        }
    }
}

impl std::fmt::Display for ValueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.casa_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every CASA type and alias from `daskms/columns.py:15-54` must map to
    /// the numpy dtype dask-ms expects.
    #[test]
    fn full_alias_to_numpy_mapping() {
        let cases: &[(&[&str], &str)] = &[
            (&["BOOL", "BOOLEAN"], "bool"),
            (&["BYTE", "UCHAR"], "uint8"),
            (&["SHORT", "SMALLINT"], "int16"),
            (&["USHORT", "USMALLINT"], "uint16"),
            (&["INT", "INTEGER"], "int32"),
            (&["UINT", "UINTEGER"], "uint32"),
            (&["FLOAT"], "float32"),
            (&["DOUBLE"], "float64"),
            (&["FCOMPLEX", "COMPLEX"], "complex64"),
            (&["DCOMPLEX"], "complex128"),
            (&["STRING"], "object"),
        ];
        for (aliases, numpy) in cases {
            for alias in *aliases {
                let vt = ValueType::from_casa_name(alias)
                    .unwrap_or_else(|_| panic!("failed to parse {alias}"));
                assert_eq!(vt.numpy_name(), *numpy, "wrong numpy dtype for {alias}");
            }
        }
    }

    #[test]
    fn aliases_are_case_insensitive() {
        assert_eq!(
            ValueType::from_casa_name("dcomplex"),
            ValueType::from_casa_name("DComplex")
        );
        assert_eq!(
            ValueType::from_casa_name("boolean").unwrap(),
            ValueType::Bool
        );
    }

    #[test]
    fn unknown_names_are_rejected() {
        assert_eq!(
            ValueType::from_casa_name("LARGEINT"),
            Err(UnknownValueType("LARGEINT".into()))
        );
        assert!(ValueType::from_casa_name("").is_err());
    }

    #[test]
    fn canonical_names_round_trip() {
        for vt in ValueType::ALL {
            assert_eq!(ValueType::from_casa_name(vt.casa_name()), Ok(vt));
        }
    }

    #[test]
    fn table_dat_names_round_trip() {
        for vt in ValueType::ALL {
            assert_eq!(ValueType::from_casa_name(vt.table_dat_name()), Ok(vt));
        }
    }

    #[test]
    fn element_sizes() {
        assert_eq!(ValueType::Bool.element_size(), Some(1));
        assert_eq!(ValueType::Byte.element_size(), Some(1));
        assert_eq!(ValueType::Short.element_size(), Some(2));
        assert_eq!(ValueType::UShort.element_size(), Some(2));
        assert_eq!(ValueType::Int.element_size(), Some(4));
        assert_eq!(ValueType::UInt.element_size(), Some(4));
        assert_eq!(ValueType::Float.element_size(), Some(4));
        assert_eq!(ValueType::Double.element_size(), Some(8));
        assert_eq!(ValueType::Complex.element_size(), Some(8));
        assert_eq!(ValueType::DComplex.element_size(), Some(16));
        assert_eq!(ValueType::String.element_size(), None);
    }

    #[test]
    fn all_types_have_unique_canonical_names() {
        let mut names: Vec<_> = ValueType::ALL.iter().map(|v| v.casa_name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ValueType::ALL.len());
    }
}
