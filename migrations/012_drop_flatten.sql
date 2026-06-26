-- flatten is now a purely UI/config-level setting, not a database column.
ALTER TABLE folders DROP COLUMN flatten;
