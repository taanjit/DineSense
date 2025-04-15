create database Dine_sense;
use Dine_sense;

create table Segmentation(GenderRatio int,
						AgeGroup int);
                        
create table WaitingTime(TotalNum int,
							MenCount int,
                            WomenCount int);

create table TableUtilization(WaiterEngageTime time,
								TimeToOrder time,
                                TimeToFood time,
                                TimeToBill time,
                                AvgEngageTime time);
#drop table Heatmap;
create table Heatmap(AvgSector1 int,
						AvgSector2 int, 
                        AvgSector3 int,
                        AvgSector4 int);
                        
create table TemperatureRec(TempSec1 varchar(10),
							TempSec2 varchar(10),
                            TempSec3 varchar(10),
                            TempSec4 varchar(10));

describe Segmentation;
describe WaitingTime;
describe TableUtilization;
describe Heatmap;
describe TemperatureRec;
								
                            
                            