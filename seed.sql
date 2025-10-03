-- school_timetable_schema_seed.sql placeholder will be overwritten below
-- school_timetable_schema_seed.sql (esteso + capacità per periodi)
BEGIN;
DROP TABLE IF EXISTS schedule;
DROP TABLE IF EXISTS class_date_capacity_periods;
DROP TABLE IF EXISTS class_day_capacity;
DROP TABLE IF EXISTS class_stage_periods;
DROP TABLE IF EXISTS teacher_unavailability_periods;
DROP TABLE IF EXISTS teacher_availability;
DROP TABLE IF EXISTS class_subject_plan;
DROP TABLE IF EXISTS merge_members;
DROP TABLE IF EXISTS merge_groups;
DROP TABLE IF EXISTS teacher_subjects;
DROP TABLE IF EXISTS holidays;
DROP TABLE IF EXISTS teachers;
DROP TABLE IF EXISTS subjects;
DROP TABLE IF EXISTS classes;

CREATE TABLE classes (
  class_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  start_date DATE
);

CREATE TABLE subjects (
  subject_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  block_len_min INT NOT NULL DEFAULT 1,
  macro_preferred BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE teachers (
  teacher_id TEXT PRIMARY KEY,
  name TEXT NOT NULL
);

CREATE TABLE teacher_subjects (
  teacher_id TEXT REFERENCES teachers(teacher_id) ON DELETE CASCADE,
  subject_id TEXT REFERENCES subjects(subject_id) ON DELETE CASCADE,
  PRIMARY KEY (teacher_id, subject_id)
);

CREATE TABLE merge_groups (
  merge_id TEXT PRIMARY KEY,
  name TEXT NOT NULL
);

CREATE TABLE merge_members (
  merge_id TEXT REFERENCES merge_groups(merge_id) ON DELETE CASCADE,
  class_id TEXT REFERENCES classes(class_id) ON DELETE CASCADE,
  PRIMARY KEY (merge_id, class_id)
);

CREATE TABLE class_subject_plan (
  plan_id TEXT PRIMARY KEY,
  class_id TEXT REFERENCES classes(class_id),
  merge_id TEXT REFERENCES merge_groups(merge_id),
  subject_id TEXT REFERENCES subjects(subject_id) NOT NULL,
  school_year TEXT,
  total_hours INT NOT NULL CHECK (total_hours > 0),
  CHECK ((class_id IS NOT NULL) <> (merge_id IS NOT NULL))
);

CREATE TABLE teacher_availability (
  teacher_id TEXT REFERENCES teachers(teacher_id) ON DELETE CASCADE,
  weekday SMALLINT NOT NULL CHECK (weekday BETWEEN 0 AND 6),
  slot_label TEXT NOT NULL CHECK (slot_label IN ('08:00','09:00','10:00','11:00','12:00','13:00','14:30','15:30')),
  PRIMARY KEY (teacher_id, weekday, slot_label)
);

CREATE TABLE teacher_unavailability_periods (
  id BIGSERIAL PRIMARY KEY,
  teacher_id TEXT REFERENCES teachers(teacher_id) ON DELETE CASCADE,
  start_date DATE NOT NULL,
  end_date DATE NOT NULL,
  CHECK (start_date <= end_date)
);

CREATE TABLE holidays ( dt DATE PRIMARY KEY );

CREATE TABLE class_stage_periods (
  id BIGSERIAL PRIMARY KEY,
  class_id TEXT REFERENCES classes(class_id) ON DELETE CASCADE,
  start_date DATE NOT NULL,
  end_date DATE NOT NULL,
  CHECK (start_date <= end_date)
);

CREATE TABLE class_day_capacity (
  class_id TEXT REFERENCES classes(class_id) ON DELETE CASCADE,
  weekday SMALLINT NOT NULL CHECK (weekday BETWEEN 0 AND 6),
  max_hours INT NOT NULL CHECK (max_hours >= 0 AND max_hours <= 8),
  PRIMARY KEY (class_id, weekday)
);

CREATE TABLE class_date_capacity_periods (
  id BIGSERIAL PRIMARY KEY,
  class_id TEXT REFERENCES classes(class_id) ON DELETE CASCADE,
  start_date DATE NOT NULL,
  end_date DATE NOT NULL,
  max_hours INT NOT NULL CHECK (max_hours >= 0 AND max_hours <= 8),
  CHECK (start_date <= end_date)
);

CREATE TABLE schedule (
  schedule_id BIGSERIAL PRIMARY KEY,
  date DATE NOT NULL,
  start_slot TEXT NOT NULL,
  end_slot TEXT NOT NULL,
  class_or_group_type CHAR(1) NOT NULL CHECK (class_or_group_type IN ('C','G')),
  class_id TEXT REFERENCES classes(class_id),
  merge_id TEXT REFERENCES merge_groups(merge_id),
  subject_id TEXT REFERENCES subjects(subject_id) NOT NULL,
  teacher_id TEXT REFERENCES teachers(teacher_id) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (
    (class_or_group_type='C' AND class_id IS NOT NULL AND merge_id IS NULL) OR
    (class_or_group_type='G' AND merge_id IS NOT NULL AND class_id IS NULL)
  )
);

-- Dati demo
INSERT INTO classes (class_id,name,start_date) VALUES
  ('1B','1B Sala Bar',NULL),
  ('1C','1C Cucina','2025-10-07');

INSERT INTO subjects (subject_id,name,block_len_min,macro_preferred) VALUES
  ('ITA','Italiano',1,false),
  ('MAT','Matematica',1,false),
  ('CUC','Laboratorio Cucina',2,true),
  ('SAL','Laboratorio Sala Bar',2,true);

INSERT INTO teachers (teacher_id,name) VALUES
  ('ROS','Rossi'),('BIA','Bianchi'),('VER','Verdi'),('NER','Neri');

INSERT INTO teacher_subjects (teacher_id,subject_id) VALUES
  ('ROS','ITA'),('ROS','MAT'),('BIA','CUC'),('VER','SAL'),('NER','MAT');

INSERT INTO merge_groups (merge_id,name) VALUES ('GBC','1B+1C Basi');
INSERT INTO merge_members (merge_id,class_id) VALUES ('GBC','1B'),('GBC','1C');

INSERT INTO class_subject_plan (plan_id,merge_id,subject_id,school_year,total_hours) VALUES
  ('P1','GBC','ITA','2025-26',3),
  ('P2','GBC','MAT','2025-26',3);
INSERT INTO class_subject_plan (plan_id,class_id,subject_id,school_year,total_hours) VALUES
  ('P3','1C','CUC','2025-26',4),
  ('P4','1B','SAL','2025-26',4);

INSERT INTO teacher_availability (teacher_id,weekday,slot_label)
SELECT t.teacher_id, wd, s.slot_label
FROM (VALUES ('ROS'),('BIA'),('VER'),('NER')) AS t(teacher_id)
CROSS JOIN generate_series(0,4) AS wd
CROSS JOIN (VALUES ('08:00'),('09:00'),('10:00'),('11:00'),('12:00'),('13:00')) AS s(slot_label);

DELETE FROM teacher_availability WHERE teacher_id='BIA' AND weekday=2 AND slot_label='10:00';
DELETE FROM teacher_availability WHERE teacher_id='VER' AND weekday=3 AND slot_label='12:00';

INSERT INTO teacher_unavailability_periods (teacher_id,start_date,end_date) VALUES
  ('BIA','2025-10-08','2025-10-08');

INSERT INTO class_stage_periods (class_id,start_date,end_date) VALUES
  ('1B','2025-10-10','2025-10-10');

INSERT INTO class_day_capacity (class_id,weekday,max_hours) VALUES ('1B',2,4);

INSERT INTO class_date_capacity_periods (class_id,start_date,end_date,max_hours) VALUES
  ('1C','2025-10-13','2025-10-17',3);
INSERT INTO class_date_capacity_periods (class_id,start_date,end_date,max_hours) VALUES
  ('1B','2025-10-14','2025-10-14',2);

COMMIT;
